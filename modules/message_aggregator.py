from collections import defaultdict
from typing import Optional
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_add, scatter_mean


class MessageAggregator(torch.nn.Module):
    """
    Abstract class for the message aggregator module, which given a batch of node ids and
    corresponding messages, aggregates messages with the same node id.
    """
    def __init__(self, device):
        super(MessageAggregator, self).__init__()
        self.device = device

    def aggregate(self, node_ids, messages):
        """
        Given a list of node ids, and a list of messages of the same length, aggregate different
        messages for the same id using one of the possible strategies.
        :param node_ids: A list of node ids of length batch_size
        :param messages: A tensor of shape [batch_size, message_length]
        :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
        """

    def group_by_id(self, node_ids, messages, timestamps):
        node_id_to_messages = defaultdict(list)
        for i, node_id in enumerate(node_ids):
            node_id_to_messages[node_id].append((messages[i], timestamps[i]))
        return node_id_to_messages

    # FIX 1 (shared utility): deterministic deduplication that respects messages dict safely.
    # Using .get() avoids the defaultdict side-effect of creating empty entries on key lookup,
    # and dict.fromkeys preserves insertion order (unlike set()) so aggregated rows always align
    # with the returned unique_nodes list.
    def _get_valid_unique_nodes(self, node_ids, messages):
        """
        Return an ordered, deduplicated list of node_ids that have at least one message.
        - Uses dict.fromkeys for deterministic ordering (set() is non-deterministic).
        - Uses messages.get() to avoid defaultdict side-effects.
        """
        seen = dict.fromkeys(node_ids)          # preserves first-seen order, O(n)
        return [n for n in seen if len(messages.get(n) or []) > 0]


class LastMessageAggregator(MessageAggregator):
    def __init__(self, device):
        super(LastMessageAggregator, self).__init__(device)

    def aggregate(self, node_ids, messages):
        """Only keep the last message for each node"""
        # FIX 1 applied: deterministic ordering + safe defaultdict access
        unique_node_ids = self._get_valid_unique_nodes(node_ids, messages)
        unique_messages = []
        unique_timestamps = []
        to_update_node_ids = []

        for node_id in unique_node_ids:
            to_update_node_ids.append(node_id)
            unique_messages.append(messages[node_id][-1][0])
            unique_timestamps.append(messages[node_id][-1][1])

        unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

        return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
    def __init__(self, device):
        super().__init__(device)

    def aggregate(self, node_ids, messages):
        # FIX 1 applied: deterministic ordering + safe defaultdict access
        unique_nodes = self._get_valid_unique_nodes(node_ids, messages)

        if len(unique_nodes) == 0:
            return [], torch.empty((0,)), torch.empty((0,))

        all_msgs = []
        node_index = []
        timestamps = []

        for idx, node in enumerate(unique_nodes):
            node_msgs = messages[node]
            for msg, ts in node_msgs:
                all_msgs.append(msg)
                node_index.append(idx)
            timestamps.append(node_msgs[-1][1])

        msg_tensor = torch.stack(all_msgs).to(self.device)
        node_index = torch.tensor(node_index, device=self.device)

        aggregated = scatter_mean(
            msg_tensor,
            node_index,
            dim=0,
            dim_size=len(unique_nodes)
        )

        timestamps = torch.stack(timestamps).to(self.device)
        return unique_nodes, aggregated, timestamps


class WeightedMessageAggregator(MessageAggregator):
   def __init__(self, message_dim, device):
        super().__init__(device)
        self.norm = nn.LayerNorm(message_dim)
        # CHANGE FROM: single linear scorer
        # TO: two-layer MLP scorer — gives the scorer enough capacity
        # to learn non-linear separation between signal and junk
        self.scorer = nn.Sequential(
            nn.Linear(message_dim, message_dim // 2),
            nn.ReLU(),
            nn.Linear(message_dim // 2, 1)
        )
     def aggregate(self, node_ids, messages):
        unique_nodes = self._get_valid_unique_nodes(node_ids, messages)

        if len(unique_nodes) == 0:
            return [], torch.empty((0,), device=self.device), torch.empty((0,), device=self.device)

        all_msgs = []
        node_index = []
        timestamps = []

        for idx, node in enumerate(unique_nodes):
            node_msgs = messages[node]
            msg_stack = torch.stack([m[0] for m in node_msgs]).to(self.device)
            all_msgs.append(msg_stack)
            node_index.append(torch.full((len(node_msgs),), idx, device=self.device))
            timestamps.append(node_msgs[-1][1])

        msg_tensor = torch.cat(all_msgs, dim=0)
        node_index = torch.cat(node_index, dim=0)

        # Weighted aggregation
        scores = self.scorer(self.norm(msg_tensor)).squeeze(-1)
        weights = scatter_softmax(scores, node_index)
        weighted_msgs = msg_tensor * weights.unsqueeze(-1)
        aggregated_weighted = scatter_add(
            weighted_msgs, node_index, dim=0, dim_size=len(unique_nodes)
        )

        # ADDED: residual mean — blends weighted result with simple mean
        # so early training is stable while scorer weights are random
        aggregated_mean = scatter_mean(
            msg_tensor, node_index, dim=0, dim_size=len(unique_nodes)
        )
        aggregated = 0.8 * aggregated_weighted + 0.2 * aggregated_mean

        timestamps = torch.stack(timestamps).to(self.device)
        return unique_nodes, aggregated, timestamps
        

class AttentionMessageAggregator(MessageAggregator):
    def __init__(self, device, n_heads: int, message_dim: int, dropout: float = 0,
                 post_norm: Optional[bool] = None, learnable: Optional[bool] = None,
                 add_cls_token: Optional[bool] = None):
        super(AttentionMessageAggregator, self).__init__(device)
        self.n_heads = n_heads
        self.message_dim = message_dim
        self.q_proj = nn.Linear(message_dim, message_dim) if learnable else nn.Identity()
        self.k_proj = nn.Linear(message_dim, message_dim) if learnable else nn.Identity()
        self.v_proj = nn.Linear(message_dim, message_dim) if learnable else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(message_dim) if post_norm else nn.Identity()
        self.add_cls_token = add_cls_token

    def attention(self, padding_message: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        num_unique_nodes, padding_length, message_dim = padding_message.shape
        head_dimension = message_dim // self.n_heads
        if head_dimension * self.n_heads != message_dim:
            raise ValueError(
                f"n_heads must divide message_dim evenly, "
                f"got message_dim={message_dim}, n_heads={self.n_heads}"
            )

        Q = self.q_proj(padding_message).view(num_unique_nodes, padding_length, self.n_heads, head_dimension).permute(0, 2, 1, 3)
        K = self.k_proj(padding_message).view(num_unique_nodes, padding_length, self.n_heads, head_dimension).permute(0, 2, 3, 1)
        V = self.v_proj(padding_message).view(num_unique_nodes, padding_length, self.n_heads, head_dimension).permute(0, 2, 1, 3)

        scale = 1.0 / torch.sqrt(torch.tensor(head_dimension, dtype=torch.float32))
        attn = torch.matmul(Q, K) * scale  # [N, heads, seq, seq]

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn = attn.masked_fill(mask == 1, -1e9)
            else:
                raise ValueError("attention_mask must have shape [num_unique_nodes, padding_length]")

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V).permute(0, 2, 1, 3)  # [N, seq, heads, head_dim]
        output = output.reshape(num_unique_nodes, padding_length, message_dim)
        output = self.layer_norm(output)
        return output

    def aggregate(self, node_ids, messages):
        # FIX 1 applied: deterministic ordering + safe defaultdict access
        valid_node_ids = self._get_valid_unique_nodes(node_ids, messages)
        message_dim = self.message_dim
        unique_timestamps = []
        unique_messages = []
        to_update_node_ids = []

        num_unique_nodes = len(valid_node_ids)

        if num_unique_nodes == 0:
            return to_update_node_ids, unique_messages, unique_timestamps

        length_message = [len(messages[nid]) for nid in valid_node_ids]
        max_length = max(length_message)

        attention_mask = torch.zeros(num_unique_nodes, max_length, device=self.device)
        padding_message = torch.zeros(num_unique_nodes, max_length, message_dim, device=self.device)

        for idx, node_id in enumerate(valid_node_ids):
            to_update_node_ids.append(node_id)
            length_per_message = len(messages[node_id])
            attention_mask[idx, length_per_message:] = 1
            padding_message[idx, :length_per_message] = torch.stack(
                [m[0] for m in messages[node_id]], dim=0
            )
            unique_timestamps.append(messages[node_id][-1][1])

        if self.add_cls_token:
            cls_message = torch.zeros(num_unique_nodes, 1, message_dim, device=self.device)
            cls_mask = torch.zeros(num_unique_nodes, 1, device=self.device)
            padding_message_with_cls = torch.cat((cls_message, padding_message), dim=1)
            attention_mask_with_cls = torch.cat((cls_mask, attention_mask), dim=1)
            updated = self.attention(padding_message_with_cls, attention_mask_with_cls)
            unique_messages = updated[:, 0, :].squeeze(1)
        else:
            updated = self.attention(padding_message, attention_mask)
            unique_messages = F.avg_pool1d(
                updated.transpose(-1, -2),
                kernel_size=max_length,
                stride=max_length
            ).squeeze(-1)

        unique_timestamps = torch.stack(unique_timestamps) if to_update_node_ids else []
        return to_update_node_ids, unique_messages, unique_timestamps


def get_message_aggregator(aggregator_type, device, n_heads, message_dim,
                            learnable, add_cls_token, dropout=0.0, post_norm=False):
    if aggregator_type == "last":
        return LastMessageAggregator(device=device)
    elif aggregator_type == "mean":
        return MeanMessageAggregator(device=device)
    elif aggregator_type == "weightedmean":
        return WeightedMessageAggregator(device=device, message_dim=message_dim)
    elif aggregator_type == "attention":
        return AttentionMessageAggregator(
            device=device, n_heads=n_heads, message_dim=message_dim,
            dropout=dropout, post_norm=post_norm,
            learnable=learnable, add_cls_token=add_cls_token
        )
    else:
        raise ValueError(f"Message aggregator {aggregator_type} not implemented")