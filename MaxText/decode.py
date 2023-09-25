"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
"""Training loop and Decoding of the model."""
import functools
from typing import Sequence

import os
from absl import app
from flax.linen import partitioning as nn_partitioning
import numpy as np
import optax

from layers import Transformer
import pyconfig
from input_pipeline import get_datasets
from input_pipeline import preprocess_dataset
import max_utils
import temperature_sampler
import orbax.checkpoint as ocp

import checkpointing

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
from dataclasses import dataclass

from jax.experimental.compilation_cache import compilation_cache as cc

import max_logging
from llama import LlamaForCausalLM

from typing import Optional
from config import LlamaConfig
from clu import parameter_overview
from transformers import AutoTokenizer, LlamaTokenizer

cc.initialize_cache(os.path.expanduser("~/jax_cache"))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

jax.numpy.set_printoptions(precision=10)

# this now works with the llama hf tokenizer, but should work with any
def decode_tokens(toks, tokenizer, eos_id):
  if np.argmax(toks == eos_id) > 0:
    valid_toks = toks[:np.argmax(toks == eos_id)]
  else:
    valid_toks = toks
    valid_toks[-1] = eos_id

  valid_toks = valid_toks.astype(np.int32)

  return tokenizer.decode(valid_toks), len(valid_toks)


def encode_strings(strs, max_len, tokenizer):
  tokenized_batch = np.zeros((len(strs), max_len), np.int32)
  for i, s in enumerate(strs):
    toks = tokenizer.encode(s, return_tensors='np')[0]
    # Remove EOS token in prompt.
    tokenized_batch[i, :toks.shape[0]-1] = toks[:-1]
  return tokenized_batch

def predict_step(inputs,
                 state,
                 rngkey,
                 model,
                 config):
  """Predict language model on a batch."""
  # NOTE: wtf are we adding inputs.shape[2:] here?  it's almost always empty??
  target_shape = (inputs.shape[0], config.max_predict_length) + inputs.shape[2:]

  initial_variables = model.init(
      rngkey, #jax.random.PRNGKey(0),
      jnp.ones(target_shape, config.dtype),
      #None,
      #enable_dropout=False,
      #decode=True,
      #max_decode_length=config.max_predict_length
  )
  #cache = initial_variables["cache"]

  def tokens_ids_to_logits(flat_ids):
    """Token slice to logits from decoder model."""

    # --> [batch * beam, 1, vocab]
    flat_logits = model.apply(
        {
            "params": state.params,
        },
        flat_ids,
        )
    #new_flat_cache = new_vars["cache"]
    # Remove singleton sequence-length dimension:
    # [batch, 1, vocab] --> [batch, vocab]
    print('we out')
    if flat_logits.shape[1] == 1:
        flat_logits = flat_logits.squeeze(axis=1)
    #new_flat_cache = None
    return flat_logits

  # Using the above-defined single-step decoder function, run a
  # search over possible sequences given input encoding.

  #print('step that errors')

  #flat_logits =  model.apply(
  #{
  #  "params": state.params,
  #},
  #inputs)

  #print(flat_logits)
  print('passing to temperature sample')
  print(inputs)

  seqs = temperature_sampler.temperature_sample(
      inputs,
      #cache,
      tokens_ids_to_logits,
      rngkey,
      temperature=config.sampling_temperature,
      topk=config.sampling_top_k,
      eos_token=config.eos_id)

  return seqs

def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
  return optax.linear_schedule(
          init_value=0,
          end_value=learning_rate,
          transition_steps=warmup_steps
          )


def decode_loop(config, state=None):
  """Decoding loop for the Transformer model."""
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(config.checkpoint_dir,
                                                                     config.enable_checkpointing,
                                                                     config.async_checkpointing)
  rng = random.PRNGKey(0)

  # Model and Optimizer definition
  #model = Transformer(config)
  model = LlamaForCausalLM(LlamaConfig(
        vocab_size= 32000,
        hidden_size= 4096,
        intermediate_size= 11008,
        num_hidden_layers=32,
        num_attention_heads= 32, 
        num_key_value_heads = 32,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5))

  """

  model = LlamaForCausalLM(LlamaConfig(
        vocab_size= 32000,
        hidden_size= 128,
        intermediate_size= 256,
        num_hidden_layers=3,
        num_attention_heads= 32, 
        num_key_value_heads = 32,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5))

  """
  tx = optax.adam(
    max_utils.create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
    )
  ) # TODO: we need an optax.GradientTransformation to form a TrainState, but we don't use it when decoding

  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Set up datasets.
  #train_ds, eval_ds = get_datasets(
  #    config=config,
  #)

  #_, _, _, sp_tokenizer = preprocess_dataset(
  #  config,
  #  mesh,
  #  train_ds, eval_ds,
  #  vocab_path=os.path.join(config.base_output_directory, config.vocab_relative_path),
  #)

  state, state_mesh_annotations = max_utils.setup_initial_state(model, tx, config, rng, mesh, checkpoint_manager)

  from jax.tree_util import tree_flatten, tree_unflatten
    
  checkpointer = ocp.PyTreeCheckpointer()
  params = checkpointer.restore('hermes')['params']
    
  temp_state = state.params
  temp_state['lm_head']['kernel'] = params['lm_head']['kernel']
  temp_state['model']['embed_tokens']['embedding'] = params['model']['embed_tokens']['embedding'].T
  temp_state['model']['norm']['weight'] = params['model']['norm']['weight'].T


  for i in range(32):
    temp_state['model'][f'layers_{i}']['input_layernorm']['weight'] = params['model'][f'layers_{i}']['input_layernorm'] 
    temp_state['model'][f'layers_{i}']['post_attention_layernorm']['weight'] = params['model'][f'layers_{i}']['post_attention_layernorm']

    temp_state['model'][f'layers_{i}']['self_attn']['q_proj']['kernel'] = params['model'][f'layers_{i}']['self_attn']['q_proj']['kernel']
    temp_state['model'][f'layers_{i}']['self_attn']['k_proj']['kernel'] = params['model'][f'layers_{i}']['self_attn']['k_proj']['kernel']
    temp_state['model'][f'layers_{i}']['self_attn']['v_proj']['kernel'] = params['model'][f'layers_{i}']['self_attn']['v_proj']['kernel']
    temp_state['model'][f'layers_{i}']['self_attn']['o_proj']['kernel'] = params['model'][f'layers_{i}']['self_attn']['o_proj']['kernel']

    temp_state['model'][f'layers_{i}']['mlp']['gate_proj']['kernel'] = params['model'][f'layers_{i}']['mlp']['gate_proj']['kernel'] 
    temp_state['model'][f'layers_{i}']['mlp']['down_proj']['kernel'] = params['model'][f'layers_{i}']['mlp']['down_proj']['kernel'] 
    temp_state['model'][f'layers_{i}']['mlp']['up_proj']['kernel'] = params['model'][f'layers_{i}']['mlp']['up_proj']['kernel'] 

  state = state.replace(params=temp_state)

  p_predict_step = pjit(
      functools.partial(predict_step, model=model, config=config),
      in_shardings=(P(None, None),
                        state_mesh_annotations,
                              None),
      out_shardings=P(None, None)
  )

  # Encode the demo prompt.
  # use transformers to encode here.
  #hf_tokenizer = LlamaTokenizer.from_pretrained('NousResearch/Nous-Hermes-llama-2-7b')
  sp_tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=True)

  tokenized_prompts = encode_strings(
      [config.prompt], config.max_predict_length, sp_tokenizer)

  tokenized_prompts = jnp.array([[1, 474, 5360, 304, 29871]], dtype=jnp.int32)

  print('no padding being used')
  print(f'encode shape: {tokenized_prompts.shape}')

  if config.metrics_file:
    local_metrics_file = open(config.metrics_file, 'a', encoding="utf8")
    metrics= {'scalar': {} }
  #max_utils.activate_profiler(config)
  for step in np.arange(20):
    rng, rng_to_use = jax.random.split(rng)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      seqs = p_predict_step(tokenized_prompts, state, rng_to_use)
      decoded_string, num_tokens_decoded = decode_tokens(np.array(seqs)[0], sp_tokenizer, 2)
      max_logging.log(f"Decoding #{step} (num tokens {num_tokens_decoded}):\n\t{decoded_string}")
      #if config.metrics_file:
      #  metrics['scalar']['num_tokens'] = num_tokens_decoded
      #  max_utils.write_metrics_locally(metrics, step, config, local_metrics_file)
  #max_utils.deactivate_profiler(config)



def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  os.environ["TFDS_DATA_DIR"] = pyconfig.config.dataset_path
  decode_loop(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
