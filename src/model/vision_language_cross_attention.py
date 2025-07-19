#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: vision_language_cross_attention
Author: Waver
"""
import numpy as np
import torch
from torch import nn
from torch.nn import init


class VisionLanguageCrossAttention(nn.Module):
    def __init__(self, logger, v_dim, l_dim=768, num_heads=8, dropout=0.0,
                 num_mem=1, num_neg_mem=1):
        super().__init__()

        self.logger = logger
        self.num_mem = num_mem
        self.num_neg_mem = num_neg_mem
        if num_mem > 0:
            self.memory_token = nn.Embedding(num_mem, v_dim)
            nn.init.xavier_normal_(self.memory_token.weight)  # 添加初始化

        if num_neg_mem > 0:
            self.neg_memory_token = nn.Embedding(num_neg_mem, v_dim)

        self.input_proj = nn.Sequential(
            nn.Conv1d(v_dim, v_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(v_dim),
        )
        self.lan_proj = nn.Sequential(
            nn.Conv1d(l_dim, v_dim, kernel_size=1, stride=1),
        )

        self.vision_lan_fuse = ScaledDotProductAttention(self.logger, v_dim, h=num_heads, dropout=dropout)
        self.memory_fuse = ScaledDotProductAttention(self.logger, v_dim, h=num_heads, dropout=dropout)
        self.feature_fuse = ScaledDotProductAttention(self.logger, v_dim, h=num_heads, dropout=dropout)
        self.norms = nn.ModuleList()
        self.norms.append(nn.InstanceNorm1d(v_dim))
        self.norms.append(nn.InstanceNorm1d(v_dim))
        self.norms.append(nn.InstanceNorm1d(v_dim))

        self.output_proj = nn.Sequential(
            nn.Conv1d(v_dim, v_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(v_dim)
        )

        self._reset_parameters()

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # nn.init.zeros_(self.output_proj[1].weight)
        # nn.init.zeros_(self.neg_memory_token.weight)

        if isinstance(self.output_proj[1], nn.BatchNorm1d):
            nn.init.ones_(self.output_proj[1].weight)
            nn.init.zeros_(self.output_proj[1].bias)

        if hasattr(self, 'neg_memory_token'):
             nn.init.normal_(self.neg_memory_token.weight, std=0.02)

    def forward(self, src, lan, pos_embed, lan_attmask):

        if src is None:
            return None, None, None, None, None

        if lan_attmask is None and lan is not None:
            lan_attmask = torch.ones(lan.size(0), lan.size(1), device=lan.device)

        B, HW, C_src = src.shape
        _, Nl, C_lan = lan.shape

        # src: (B, HW, C_src) -> (B, C_src, HW) for Conv1d
        src_permuted = src.permute(0, 2, 1)
        src_projected = self.input_proj(src_permuted)  # Output: (B, v_dim, HW)

        # lan: (B, Nl, C_lan) -> (B, C_lan, Nl) for Conv1d
        lan_permuted = lan.permute(0, 2, 1)
        lan_projected = self.lan_proj(lan_permuted)  # Output: (B, v_dim, Nl)

        # Permute back for Attention: (B, SeqLen, FeatureDim)
        # src_for_attn: (B, HW, v_dim)
        # lan_for_attn: (B, Nl, v_dim)
        src_for_attn = src_projected.permute(0, 2, 1)
        lan_for_attn = lan_projected.permute(0, 2, 1)

        # lan_attmask for Attention needs to be (B, Nl, 1) or broadcastable
        if lan_attmask.ndim == 2:  # If it was (B, Nl)
            lan_attmask_for_mhsa = lan_attmask.unsqueeze(2)  # (B, Nl, 1)
        elif lan_attmask.ndim == 3 and lan_attmask.shape[2] == 1:  # Already (B, Nl, 1)
            lan_attmask_for_mhsa = lan_attmask
        else:
            # Fallback, but this indicates an issue
            lan_attmask_for_mhsa = torch.ones(lan_for_attn.size(0), lan_for_attn.size(1), 1, device=lan_for_attn.device)

        # --- 2. Vision-Language Fusion (vision_lan_fuse) ---
        vision_lan_fused_output, vision_lan_att = self.vision_lan_fuse(lan_for_attn, src_for_attn, src_for_attn,
                                                                       query_mask=lan_attmask_for_mhsa)

        # Norm 0 (InstanceNorm1d expects (B, C, L))
        vision_lan_fused_permuted = vision_lan_fused_output.permute(0, 2, 1)
        norm0_output = self.norms[0](vision_lan_fused_permuted)
        vision_lan_fused_normed = norm0_output.permute(0, 2, 1)  # Back to (B, Nl, C)


        # --- 3. Memory Fusion (memory_fuse) ---
        current_lan_mem = lan_for_attn  # Start with projected language features
        mem_att = None  # Initialize

        if self.num_mem > 0:
            memory_tokens_embed = self.memory_token.weight.unsqueeze(0).repeat(B, 1, 1)  # B, num_mem, v_dim

            # Query: memory_tokens_embed
            # Key/Value: vision_lan_fused_normed (output of previous step) OR original lan_for_attn
            # Your current code uses: self.memory_fuse(memory_token, lan, lan, key_mask=lan_attmask)
            # -> lan is original projected language
            # Let's log based on current code:

            updated_memory_tokens, mem_att = \
                self.memory_fuse(memory_tokens_embed, lan_for_attn, lan_for_attn, key_mask=lan_attmask_for_mhsa)
            current_lan_mem = updated_memory_tokens  # Now lan_mem refers to the updated memory tokens

        if self.num_neg_mem > 0:
            neg_memory_tokens_embed = self.neg_memory_token.weight.unsqueeze(0).repeat(B, 1, 1)

            current_lan_mem = torch.cat([neg_memory_tokens_embed, current_lan_mem], dim=1)

        # Norm 1 (Applied if lan_mem has more than 1 token, which is usually true if num_mem or num_neg_mem > 0)
        # current_lan_mem_for_norm is (B, C, TotalNumMemOrLangTokens)
        if current_lan_mem is not None and current_lan_mem.shape[1] > 1:

            current_lan_mem_permuted = current_lan_mem.permute(0, 2, 1)
            norm1_output = self.norms[1](current_lan_mem_permuted)
            current_lan_mem_normed = norm1_output.permute(0, 2, 1)
            final_lan_mem_for_output = current_lan_mem_normed  # This will be one of the VLCA outputs
        elif current_lan_mem is not None:  # shape[1] <= 1, norm not applied
            final_lan_mem_for_output = current_lan_mem
        else:  # current_lan_mem is None
            final_lan_mem_for_output = None  # Propagate None

        # --- 4. Feature Fusion (feature_fuse) ---
        # Query: src_for_attn (projected visual features)
        # Key/Value: final_lan_mem_for_output (updated memory/language features)
        # key_mask for feature_fuse: Your current code `self.feature_fuse(src, lan_mem, lan_mem)`
        # doesn't pass a key_mask.
        # If final_lan_mem_for_output includes padding or irrelevant tokens
        # (e.g. from original language if num_mem=0), a mask might be needed.
        # For now, let's assume no key_mask or an all-ones mask if final_lan_mem_for_output is the K/V.

        # Prepare key_mask for feature_fuse if needed, based on final_lan_mem_for_output's actual content
        key_mask_for_ff = None
        if final_lan_mem_for_output is not None:
            # Example: if final_lan_mem_for_output was created by concatenating neg_mem, mem, and original_lan
            # And original_lan had lan_attmask_for_mhsa
            # This part needs careful thought based on how final_lan_mem_for_output is constructed
            # For simplicity now, assume no specific mask or an all-ones mask.
            # key_mask_for_ff = torch.ones(B, final_lan_mem_for_output.shape[1], 1, device=final_lan_mem_for_output.device)
            pass  # Current code doesn't pass key_mask to feature_fuse

        if src_for_attn is None or final_lan_mem_for_output is None:
            fused_src_output = src_for_attn  # Or None if src_for_attn is None
            feature_att = None
        else:
            fused_src_output, feature_att = self.feature_fuse(src_for_attn, final_lan_mem_for_output,
                                                              final_lan_mem_for_output, key_mask=key_mask_for_ff)

        # Norm 2
        if fused_src_output is not None:
            fused_src_permuted = fused_src_output.permute(0, 2, 1)
            norm2_output = self.norms[2](fused_src_permuted)
            fused_src_normed = norm2_output.permute(0, 2, 1)  # Back to (B, HW, C)
            src_after_ff_and_norm = fused_src_normed
        else:
            src_after_ff_and_norm = None

        # --- 5. Output Projection ---
        if src_after_ff_and_norm is not None:
            src_for_outproj_permuted = src_after_ff_and_norm.permute(0, 2, 1)  # (B, C, HW)
            final_src_output = self.output_proj(src_for_outproj_permuted).permute(0, 2, 1)  # (B, HW, C)
        else:
            final_src_output = None

        # The first returned value 'src' is used as 'fuse' in SwinTransformerStage
        # The second returned value 'lan_mem' is used as 'memory' in SwinTransformerStage
        return final_src_output, final_lan_mem_for_output, vision_lan_att, mem_att, feature_att


class ScaledDotProductAttention(nn.Module):

    def __init__(self, logger, d_model, h, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()

        self.logger = logger
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.fc_q = nn.Linear(d_model, h * self.d_k)
        self.fc_k = nn.Linear(d_model, h * self.d_k)
        self.fc_v = nn.Linear(d_model, h * self.d_v)
        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, query_mask=None, key_mask=None,
                attention_mask=None, attention_weights=None):

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        # --- Q, K, V Projections ---
        q_proj = self.fc_q(queries)
        k_proj = self.fc_k(keys)
        v_proj = self.fc_v(values)

        q = q_proj.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = k_proj.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = v_proj.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        # Masking Q, K, V (if masks are provided)
        if query_mask is not None:
            q_original_mean = q.mean().item()
            q = q * query_mask.view(b_s, nq, 1, 1).repeat(1, 1, self.h, 1).permute(0, 2, 1, 3)
        if key_mask is not None:
            k_original_mean = k.mean().item()
            v_original_mean = v.mean().item()
            k = k * key_mask.view(b_s, nk, 1, 1).repeat(1, 1, self.h, 1).permute(0, 2, 3, 1)
            v = v * key_mask.view(b_s, nk, 1, 1).repeat(1, 1, self.h, 1).permute(0, 2, 1, 3)

        # --- Attention Calculation ---
        att_raw = torch.matmul(q, k) / np.sqrt(self.d_k)

        att = att_raw # Assign to att for subsequent operations
        if attention_weights is not None:
            att_before_custom_weights = att.clone()
            att = att * attention_weights
        if attention_mask is not None: # This is the additive mask for softmax
            att_before_softmax_mask = att.clone()
            att = att.masked_fill(attention_mask, -float('inf')) # Use float('-inf') for PyTorch

        att_softmax = torch.softmax(att, -1)

        att_dropout = self.dropout(att_softmax)
        # Attention after dropout: att_dropout {get_string(att_dropout, 'mhca_att_dropout')}")

        # --- Output Calculation ---
        out_matmul = torch.matmul(att_dropout, v)
        out_reshaped = out_matmul.permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)

        out_final = self.fc_o(out_reshaped)

        return out_final, att_softmax
