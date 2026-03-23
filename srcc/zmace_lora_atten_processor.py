from diffusers.models.attention_processor import Attention
from diffusers.models.lora import LoRALinearLayer
import torch
from torch import nn


class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, attn_controller=None, module_name=None, knowledge_graph=None, entity_embedding=None) -> None:
        self.attn_controller = attn_controller
        self.module_name = module_name
        self.knowledge_graph = knowledge_graph
        self.entity_embedding = entity_embedding

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # 使用知识图谱调整注意力权重
        if self.knowledge_graph is not None and self.entity_embedding is not None:
            attention_probs = self.adjust_attention_probs(attention_probs, hidden_states, encoder_hidden_states)

        if key.shape[1] == 77 and self.attn_controller is not None:
            self.attn_controller(attention_probs, self.module_name, preserve_prior=True, latent_num=hidden_states.shape[0])

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def adjust_attention_probs(self, attention_probs, hidden_states, encoder_hidden_states):
        """
        根据知识图谱信息调整注意力权重。
        """
        # 1. 获取实体嵌入
        entity_embeddings = self.get_entity_embeddings(hidden_states, encoder_hidden_states)

        # 2. 计算注意力权重调整系数
        if entity_embeddings is not None:
            adjustment = self.calculate_adjustment(entity_embeddings, hidden_states, encoder_hidden_states)
            attention_probs = attention_probs * adjustment
        return attention_probs
    

    def get_entity_embeddings(self, hidden_states, encoder_hidden_states):
        
        if self.knowledge_graph is None or self.entity_embedding is None:
            return None

        batch_size, sequence_length, _ = hidden_states.shape
        embedding_dim = self.entity_embedding.embedding_dim

        entity_embeddings = torch.zeros((batch_size, sequence_length, embedding_dim), device=hidden_states.device)

        for i in range(batch_size):
            for j in range(sequence_length):
                # 提取 hidden_states[i, j] 对应的文本实体
                entity_name = self.extract_entity_name(hidden_states[i, j])  # 提前实现extract_entity_name 函数

                if entity_name in self.knowledge_graph:
                    entity_id = self.knowledge_graph[entity_name]
                    entity_embeddings[i, j] = self.entity_embedding(torch.tensor(entity_id).to(hidden_states.device))
                else:
                    # 如果实体不在知识图谱中，则使用零向量
                    pass

        return entity_embeddings
    

    def calculate_adjustment(self, entity_embeddings, hidden_states, encoder_hidden_states):
        """
        计算注意力权重调整系数。
        """
        if entity_embeddings is None:
            return 1.0

        batch_size, sequence_length, _ = hidden_states.shape
        num_heads = 8  # 假设有 8 个注意力头，需要根据实际情况修改

        adjustment = torch.ones((batch_size, num_heads, sequence_length, sequence_length), device=hidden_states.device)

        # 计算调整系数
        for i in range(batch_size):
            for head in range(num_heads):
                for j in range(sequence_length):
                    for k in range(sequence_length):
                        # 计算实体嵌入之间的相似度
                        similarity = torch.cosine_similarity(entity_embeddings[i, j], entity_embeddings[i, k], dim=0)

                        # 根据相似度调整注意力权重
                        adjustment[i, head, j, k] = 1.0 + similarity  # 可以使用其他函数进行调整

        return adjustment

    # def get_entity_embeddings(self, hidden_states, encoder_hidden_states):
    #     """
    #     获取实体嵌入。
    #     """
    #     # TODO: 实现获取实体嵌入的逻辑。
    #     # 可以使用 self.knowledge_graph 和 self.entity_embedding 对象。
    #     # 返回一个实体嵌入张量，形状为 (batch_size, sequence_length, embedding_dim)。
    #     return None

    # def calculate_adjustment(self, entity_embeddings, hidden_states, encoder_hidden_states):
    #     """
    #     计算注意力权重调整系数。
    #     """
    #     # TODO: 实现计算注意力权重调整系数的逻辑。
    #     # 可以使用实体嵌入、hidden_states 和 encoder_hidden_states 来计算调整系数。
    #     # 返回一个调整系数张量，形状为 (batch_size, num_heads, sequence_length, sequence_length)。
    #     return 1.0


# 创建 LoRAAttnProcessor 实例，并将其设置为 UNet 模型的注意力处理器
class LoRAAttnProcessor(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, attn_controller=None, module_name=None,
                 network_alpha=None, knowledge_graph=None, entity_embedding=None, **kwargs):
        super().__init__()

        self.attn_controller = attn_controller
        self.module_name = module_name

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        # 新增：接收 knowledge_graph 和 entity_embedding 参数
        self.knowledge_graph = knowledge_graph
        self.entity_embedding = entity_embedding

        q_rank = kwargs.pop("q_rank", None)
        q_hidden_size = kwargs.pop("q_hidden_size", None)
        q_rank = q_rank if q_rank is not None else rank
        q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size

        v_rank = kwargs.pop("v_rank", None)
        v_hidden_size = kwargs.pop("v_hidden_size", None)
        v_rank = v_rank if v_rank is not None else rank
        v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size

        out_rank = kwargs.pop("out_rank", None)
        out_hidden_size = kwargs.pop("out_hidden_size", None)
        out_rank = out_rank if out_rank is not None else rank
        out_hidden_size = out_hidden_size if out_hidden_size is not None else hidden_size

        self.to_q_lora = LoRALinearLayer(q_hidden_size, q_hidden_size, q_rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(out_hidden_size, out_hidden_size, out_rank, network_alpha)

    def __call__(self, attn: Attention, hidden_states, *args, **kwargs):

        attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
        attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
        attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
        attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

        attn._modules.pop("processor")
        attn.processor = AttnProcessor(self.attn_controller, self.module_name, self.knowledge_graph, self.entity_embedding)

        return attn.processor(attn, hidden_states, *args, **kwargs)