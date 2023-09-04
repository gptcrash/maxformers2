import torch
import os
from etils import epath
import orbax.checkpoint as ocp
import numpy as np

from typing import Optional, Any
import shutil

import numpy as np
import jax
from jax import random, numpy as jnp

import flax
from flax import linen as nn
from flax.training import checkpoints, train_state
from flax import struct, serialization
import orbax.checkpoint

import optax

TEMP_DIR = "~/.maxformers_temp"
MODEL_LINK = "https://huggingface.co/NousResearch/Nous-Hermes-llama-2-7b/resolve/main/pytorch_model.bin"
MODEL_FILENAME = MODEL_LINK.split('/')[-1]
MODEL_PATH = '/home/artem/.maxformers_temp/pytorch_model.bin'
#os.path.join(TEMP_DIR, MODEL_FILENAME)
#os.system(f'wget {MODEL_LINK} -P {TEMP_DIR}')

pytree = {}


# open as a state_dict
state_dict = torch.load(os.path.abspath(MODEL_PATH))

for k, v in state_dict.items():
    print(f'{k} {v.shape}')

# get lm head
pytree['lm_head'] = {}
pytree['lm_head']['kernel'] = state_dict['lm_head.weight']

pytree['model'] = {}
pytree['model']['embed_tokens'] = {}
pytree['model']['embed_tokens']['embedding'] = state_dict['model.embed_tokens.weight']

pytree['model']['norm'] = {}
pytree['model']['norm']['weight'] = state_dict['model.norm.weight']


for i in range(32):
    
    pytree['model'][f'layers_{i}'] = {}
    pytree['model'][f'layers_{i}']['input_layernorm'] = state_dict[f'model.layers.{i}.input_layernorm.weight']

    pytree['model'][f'layers_{i}']['post_attention_layernorm'] = state_dict[f'model.layers.{i}.post_attention_layernorm.weight']
    pytree['model'][f'layers_{i}']['self_attn'] = {}
    pytree['model'][f'layers_{i}']['self_attn']['q_proj'] = {}
    pytree['model'][f'layers_{i}']['self_attn']['q_proj']['kernel'] = state_dict[f'model.layers.{i}.self_attn.q_proj.weight']
    pytree['model'][f'layers_{i}']['self_attn']['k_proj'] = {}
    pytree['model'][f'layers_{i}']['self_attn']['k_proj']['kernel'] = state_dict[f'model.layers.{i}.self_attn.k_proj.weight']
    pytree['model'][f'layers_{i}']['self_attn']['v_proj'] = {}
    pytree['model'][f'layers_{i}']['self_attn']['v_proj']['kernel'] = state_dict[f'model.layers.{i}.self_attn.v_proj.weight']
    pytree['model'][f'layers_{i}']['self_attn']['o_proj'] = {}
    pytree['model'][f'layers_{i}']['self_attn']['o_proj']['kernel'] = state_dict[f'model.layers.{i}.self_attn.o_proj.weight']

    pytree['model'][f'layers_{i}']['mlp'] = {}
    pytree['model'][f'layers_{i}']['mlp']['gate_proj'] = {}
    pytree['model'][f'layers_{i}']['mlp']['gate_proj']['kernel'] = state_dict[f'model.layers.{i}.mlp.gate_proj.weight']

    pytree['model'][f'layers_{i}']['mlp']['up_proj'] = {}
    pytree['model'][f'layers_{i}']['mlp']['up_proj']['kernel'] = state_dict[f'model.layers.{i}.mlp.up_proj.weight']

    pytree['model'][f'layers_{i}']['mlp']['down_proj'] = {}
    pytree['model'][f'layers_{i}']['mlp']['down_proj']['kernel'] = state_dict[f'model.layers.{i}.mlp.down_proj.weight']


path = 'hermes'

# checkpointer = ocp.PyTreeCheckpointer()
# 'checkpoint_name' must not already exist.
# checkpointer.save(path, {'step': {}, 'params': pytree, 'count': {}})

train_state.TrainState.create(
    apply_fn= None,
    params=variables['params'],
    tx=tx)


print('saved')

#os.system(f'rm {MODEL_PATH}')

