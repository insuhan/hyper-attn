import math
import torch

from .hyper_attn import HyperAttention

# Edited from https://huggingface.co/THUDM/chatglm2-6b-32k/blob/main/modeling_chatglm.py#L194
class FastCoreAttention(torch.nn.Module):
    
    def __init__(self, config, layer_number, **kwargs):
        super(FastCoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)
        
        self.attn_method = kwargs.get('attn_method')
        if self.attn_method in ['hyper', 'hyper-cuda']:
            lsh_num_projs = kwargs.get('lsh_num_projs')
            block_size = kwargs.get('block_size')
            sample_size = kwargs.get('sample_size')
            min_seq_len = kwargs.get('min_seq_len')
            self.attn = HyperAttention(
                input_dim=128,
                lsh_num_projs=lsh_num_projs, 
                block_size=block_size,
                sample_size=sample_size, 
                min_seq_len=min_seq_len,
                cuda='cuda' in self.attn_method)
        else: 
            raise NotImplementedError("Invalid attn_method option")
        

    def forward(self,  query_layer, key_layer, value_layer, attention_mask):

        query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
        if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
            softmax_scale = query_layer.shape[-1]**(-0.5)
            context_layer = self.attn(query_layer, key_layer, value_layer, causal=True)

        else:
            assert False, 'this part the query length and key length may be different and not be a computational bottleneck.'
            if attention_mask is not None:
                attention_mask = ~attention_mask
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attention_mask)

        context_layer = context_layer.permute(2, 0, 1, 3)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer
