from dataclasses import dataclass
from typing import Tuple, Union, Optional, List

@dataclass
class LlamaConfig:
    prompt: str = "hello world"
    max_predict_length: int = 32
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = 0
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[int] = 2
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    output_attentions: bool = False
    output_hidden_states: bool = False
    output_loss: bool = False
    use_return_dict: bool = False
    dtype = 'float32'

@dataclass
class BertConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[int] = 2
    tie_word_embeddings: bool = False
    dtype = 'float32'

@dataclass
class MeshConfig:
    pass
