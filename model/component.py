# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate
@File: component.py
@Author: QI
@Date: 2021/6/7 16:44
@Description: 模型组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import args


class UpdateSpeaker(nn.Module):
    def __init__(self):
        super(UpdateSpeaker, self).__init__()
        self.utter_hidden_size = args.utter_hidden * 2
        # Parameters for speaker
        self.WA_xr = nn.Linear(self.utter_hidden_size, args.user_dim, bias=False)
        self.WA_xp = nn.Linear(self.utter_hidden_size, args.user_dim, bias=False)
        self.WA_xz = nn.Linear(self.utter_hidden_size, args.user_dim, bias=False)
        self.WA_xh = nn.Linear(self.utter_hidden_size, args.user_dim, bias=False)

        self.WA_hr = nn.Linear(args.user_dim, args.user_dim, bias=False)
        self.WA_hp = nn.Linear(args.user_dim, args.user_dim, bias=False)
        self.WA_hz = nn.Linear(args.user_dim, args.user_dim, bias=False)
        self.WA_hh = nn.Linear(args.user_dim, args.user_dim, bias=False)

        self.VA_hr = nn.Linear(args.user_dim, args.user_dim, bias=False)
        self.VA_hp = nn.Linear(args.user_dim, args.user_dim, bias=False)
        self.VA_hz = nn.Linear(args.user_dim, args.user_dim, bias=False)
        self.VA_hh = nn.Linear(args.user_dim, args.user_dim, bias=False)

    def forward(self, encoder_hidden, spk_vectors, adr_vectors):
        '''
        UpdateSpeaker
        @param encoder_hidden: [batch_size,utter_hidden*2]
        @param spk_vectors: [batch_size,user_dim]
        @param adr_vectors: [batch_size,user_dim]
        @return: new_spk_vectors: [batch_size,user_dim]
        '''
        r_t = torch.sigmoid(self.WA_xr(encoder_hidden) + self.WA_hr(spk_vectors) + self.VA_hr(adr_vectors))  # [batch_size,user_dim]
        p_t = torch.sigmoid(self.WA_xp(encoder_hidden) + self.WA_hp(spk_vectors) + self.VA_hp(adr_vectors))  # [batch_size,user_dim]
        z_t = torch.sigmoid(self.WA_xz(encoder_hidden) + self.WA_hz(spk_vectors) + self.VA_hz(adr_vectors))  # [batch_size,user_dim]

        spk_vectors_temp = torch.tanh(
            self.WA_xh(encoder_hidden) + self.WA_hh((r_t * spk_vectors)) + self.VA_hh((p_t * adr_vectors)))  # [batch_size,user_dim]
        new_spk_vectors = (1. - z_t) * spk_vectors + z_t * spk_vectors_temp  # [batch_size,user_dim]
        return new_spk_vectors


class UpdateAddressee(nn.Module):
    def __init__(self):
        super(UpdateAddressee, self).__init__()
        self.utter_hidden_size = args.utter_hidden * 2
        # Parameters for addressee
        self.WB_xr = nn.Linear(self.utter_hidden_size, args.user_dim, bias=False)
        self.WB_xp = nn.Linear(self.utter_hidden_size, args.user_dim, bias=False)
        self.WB_xz = nn.Linear(self.utter_hidden_size, args.user_dim, bias=False)
        self.WB_xh = nn.Linear(self.utter_hidden_size, args.user_dim, bias=False)

        self.WB_hr = nn.Linear(args.user_dim, args.user_dim, bias=False)
        self.WB_hp = nn.Linear(args.user_dim, args.user_dim, bias=False)
        self.WB_hz = nn.Linear(args.user_dim, args.user_dim, bias=False)
        self.WB_hh = nn.Linear(args.user_dim, args.user_dim, bias=False)

        self.VB_hr = nn.Linear(args.user_dim, args.user_dim, bias=False)
        self.VB_hp = nn.Linear(args.user_dim, args.user_dim, bias=False)
        self.VB_hz = nn.Linear(args.user_dim, args.user_dim, bias=False)
        self.VB_hh = nn.Linear(args.user_dim, args.user_dim, bias=False)

    def forward(self, encoder_hidden, adr_vectors, spk_vectors):
        '''
        UpdateAddressee
        @param encoder_hidden: [batch_size,utter_hidden*2]
        @param adr_vectors: [batch_size,user_dim]
        @param spk_vectors: [batch_size,user_dim]
        @return: new_adr_vectors
        '''
        r_t = torch.sigmoid(self.WB_xr(encoder_hidden) + self.WB_hr(adr_vectors) + self.VB_hr(spk_vectors))  # [batch_size,user_dim]
        p_t = torch.sigmoid(self.WB_xp(encoder_hidden) + self.WB_hp(adr_vectors) + self.VB_hp(spk_vectors))  # [batch_size,user_dim]
        z_t = torch.sigmoid(self.WB_xz(encoder_hidden) + self.WB_hz(adr_vectors) + self.VB_hz(spk_vectors))  # [batch_size,user_dim]
        adr_vectors_temp = torch.tanh(
            self.WB_xh(encoder_hidden) + self.WB_hh((r_t * adr_vectors)) + self.VB_hh((p_t * spk_vectors)))  # [batch_size,user_dim]
        new_adr_vectors = (1. - z_t) * adr_vectors + z_t * adr_vectors_temp
        return new_adr_vectors


class UpdateOthers(nn.Module):
    def __init__(self):
        super(UpdateOthers, self).__init__()
        self.utter_hidden_size = args.utter_hidden * 2
        # Parameters for other
        self.Wother_xr = nn.Linear(self.utter_hidden_size, args.user_dim, bias=False)
        self.Wother_xz = nn.Linear(self.utter_hidden_size, args.user_dim, bias=False)
        self.Wother_xh = nn.Linear(self.utter_hidden_size, args.user_dim, bias=False)

        self.Wother_hr = nn.Conv1d(in_channels=args.max_roles - 2, out_channels=args.max_roles - 2, kernel_size=1, bias=False)
        self.Wother_hz = nn.Conv1d(in_channels=args.max_roles - 2, out_channels=args.max_roles - 2, kernel_size=1, bias=False)
        self.Wother_hh = nn.Conv1d(in_channels=args.max_roles - 2, out_channels=args.max_roles - 2, kernel_size=1, bias=False)

    def forward(self, encoder_hidden, others_vectors):
        '''
        UpdateOthers
        @param encoder_hidden: [batch_size,utter_hidden*2]
        @param others_vectors: [batch_size,max_roles-2,user_dim]
        @return: new_others_vectors: [batch_size,max_roles-2,user_dim]
        '''
        r_t = torch.sigmoid(self.Wother_xr(encoder_hidden).unsqueeze(1).repeat(1, args.max_roles - 2, 1) +
                            self.Wother_hr(others_vectors))  # [batch_size,max_roles-2,user_dim]
        z_t = torch.sigmoid(self.Wother_xz(encoder_hidden).unsqueeze(1).repeat(1, args.max_roles - 2, 1) +
                            self.Wother_hz(others_vectors))  # [batch_size,max_roles-2,user_dim]
        others_vectors_temp = torch.tanh(self.Wother_xh(encoder_hidden).unsqueeze(1).repeat(1, args.max_roles - 2, 1) +
                                         self.Wother_hh((r_t * others_vectors)))  # [batch_size,max_roles-2,user_dim]
        new_others_vectors = (1. - z_t) * others_vectors + z_t * others_vectors_temp
        return new_others_vectors


class LuongAttention(nn.Module):
    """
    Luong attention
    """
    def __init__(self, method, hidden_size):
        super(LuongAttention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.matmul(hidden, encoder_output.permute(0, 2, 1)).squeeze(1)  # [batch,windows]

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.matmul(hidden, energy.permute(0, 2, 1)).squeeze(1)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat([hidden.repeat(1, encoder_output.size(1), 1), encoder_output], dim=-1)).tanh()
        return torch.matmul(hidden, energy.permute(0, 2, 1)).squeeze(1)

    def forward(self, hidden, encoder_outputs):
        """
        Calculate Luong attention weights
        :param hidden: [batch,1,user_dim]
        :param encoder_outputs: [batch,utter_hidden*2,windows]
        :return: attention weights
        """
        attn_energies = None
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)  # [batch_size,max_roles-1]
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # [batch_size,1,max_roles-1]


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, args.utter_hidden),
            nn.ReLU(True),
            nn.Linear(args.utter_hidden, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs: [batch_size,window,utter_hidden*2]
        energy = self.projection(encoder_outputs)  # [batch_size,window,1]
        weights = F.softmax(energy.squeeze(-1), dim=1)  # [batch_size,window]

        temp = encoder_outputs * weights.unsqueeze(-1)  # [batch_size,window,utter_hidden*2]
        outputs = temp.sum(dim=1)
        context_vector = outputs  # [batch_size,utter_hidden*2]
        return context_vector, weights
