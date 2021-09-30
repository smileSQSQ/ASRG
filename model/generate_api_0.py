# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate
@File: generate_api_0.py
@Author: QI
@Date: 2021/6/1 15:40
@Description: Original Transformer
"""

import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from config import args
from utils import Vocab, load_pickle


def src_tgt_mask(sz):
    """
    sequence mask - 防止未来信息泄露
    :param sz: padding之后的句子长度, 标量, 值为sents_max_len/target_max_len
    :return: 下三角矩阵
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(args.device)  # [sz,sz]


def src_tgt_key_padding_mask(real_length, max_len, mode="src"):
    """
    padding mask - 处理输入不定长
    :param real_length: 一个batch中每个句子的真实长度，[batch_size]
    :param max_len: 句子最大长度
    :param mode: src/tgt不用模式处理
    :return: padding_mask
    """
    padding_mask = torch.zeros([args.batch_size, max_len], dtype=torch.long).to(args.device)

    if mode == "src":
        for i in range(args.batch_size):
            no_padding = torch.zeros([1, real_length[i].item()], dtype=torch.long).to(args.device)
            padding = torch.ones([1, max_len - real_length[i].item()], dtype=torch.long).to(args.device)
            padding_mask[i, :] = torch.cat([no_padding, padding], dim=-1)
    elif mode == "tgt":
        for i in range(args.batch_size):
            no_padding = torch.zeros([1, real_length[i].item() - 1], dtype=torch.long).to(args.device)
            padding = torch.ones([1, max_len - (real_length[i].item() - 1)], dtype=torch.long).to(args.device)
            padding_mask[i, :] = torch.cat([no_padding, padding], dim=-1)

    padding_mask = padding_mask.bool()
    return padding_mask.to(args.device)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MyTransformer(nn.Module):
    def __init__(self):
        super(MyTransformer, self).__init__()

        # pretrained_embeddings = torch.cat([torch.FloatTensor(vocab.pretrained_embeddings), torch.zeros([args.vocab_size, args.d_model - args.embedding_dim])], dim=-1)
        # self.word_embedding = nn.Embedding(args.vocab_size, args.d_model).from_pretrained(pretrained_embeddings, freeze=False)
        self.word_embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.pos_embedding = PositionalEncoding(args.d_model, args.t_dropout)

        encoder_layer = TransformerEncoderLayer(args.d_model, args.n_heads, args.d_ff, args.t_dropout, "gelu")
        self.transformer_encoder = TransformerEncoder(encoder_layer, args.n_layers)

        decoder_layer = TransformerDecoderLayer(args.d_model, args.n_heads, args.d_ff, args.t_dropout, "gelu")
        self.transformer_decoder = TransformerDecoder(decoder_layer, args.n_layers)

        self.fc = nn.Linear(args.d_model, args.vocab_size)

    def forward(self, src, sents_length, tgt, response_length, A_res, A_tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        MyTransformer
        :param src: [batch_size,sents_max_len]
        :param sents_length: [batch_size]
        :param tgt: [batch_size,tgt_max_len-1]
        :param response_length: [batch_size]
        :param A_res: [batch_size,user_dim]
        :param A_tgt: [batch_size,user_dim]
        :param tgt_mask: [tgt_max_len,tgt_max_len]
        :param src_key_padding_mask: [batch_size,src_max_len]
        :param tgt_key_padding_mask: [batch_size,tgt_max_len]
        :return: generate_output
        """
        src = src.transpose(0, 1)  # [sents_max_len,batch_size]
        tgt = tgt.transpose(0, 1)
        embedded_src = self.word_embedding(src) * math.sqrt(args.d_model)
        embedded_src = self.pos_embedding(embedded_src)  # [sents_max_len,batch_size,d_model]

        src_key_padding_mask = src_tgt_key_padding_mask(sents_length, args.sents_max_len, "src")
        memory = self.transformer_encoder(embedded_src, src_key_padding_mask=src_key_padding_mask)  # [sents_max_len,batch_size,d_model]

        embedded_tgt = self.word_embedding(tgt) * math.sqrt(args.d_model)
        embedded_tgt = self.pos_embedding(embedded_tgt)  # [tgt_max_len,batch_size,d_model]

        tgt_mask = src_tgt_mask(args.target_max_len - 1)
        tgt_key_padding_mask = src_tgt_key_padding_mask(response_length, args.target_max_len - 1, "tgt")
        generate_output = self.transformer_decoder(embedded_tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)  # [tgt_max_len,batch_size,d_model]

        generate_output = self.fc(generate_output)  # [tgt_max_len,batch_size,vocab_size]
        generate_output = generate_output.permute(1, 0, 2)  # [batch_size,tgt_max_len,vocab_size]
        generate_output = generate_output.reshape(-1, generate_output.size(-1))  # [batch_size*tgt_max_len,vocab_size]

        return generate_output

    def greedy_decode(self, src, sents_length, A_res, A_tgt, start_symbol=Vocab.SOS):
        """
        Greedy Decode
        :param src: [batch_size,sents_max_len]
        :param sents_length: [batch_size]
        :param A_res: [batch_size,user_dim]
        :param A_tgt: [batch_size,user_dim]
        :param start_symbol: Vocab.SOS
        :return: dec_inputs  #[target_max_len, batch_size], batch_size=1
        """
        src = src.transpose(0, 1)  # [sents_max_len,batch_size]
        embedded_src = self.word_embedding(src) * math.sqrt(args.d_model)
        embedded_src = self.pos_embedding(embedded_src)  # [sents_max_len,batch_size,d_model]

        src_key_padding_mask = src_tgt_key_padding_mask(sents_length, args.sents_max_len)
        memory = self.transformer_encoder(embedded_src, src_key_padding_mask)

        dec_inputs = torch.zeros([args.target_max_len - 1, args.batch_size], dtype=torch.long).to(args.device)  # [target_max_len, 1]

        next_symbol = start_symbol  # SOS
        for i in range(0, args.target_max_len - 1):
            dec_inputs[i][0] = next_symbol
            cur_dec_inputs = self.word_embedding(dec_inputs) * math.sqrt(args.d_model)
            cur_dec_inputs = self.pos_embedding(cur_dec_inputs)  # [tgt_max_len,batch_size,d_model]

            dec_output = self.transformer_decoder(cur_dec_inputs, memory)  # [batch_size, tgt_len, d_model]
            dec_output = self.fc(dec_output)  # [tgt_max_len,batch_size,vocab_size]
            dec_output = dec_output.permute(1, 0, 2)  # [batch_size=1,tgt_max_len,vocab_size]

            prob = dec_output.squeeze(0).max(dim=-1, keepdim=False)[1]  # [tgt_max_len]
            next_word = prob.data[i]
            next_symbol = next_word.item()
        return dec_inputs

    def generate(self, src, sents_length, A_res, A_tgt):
        """
        Generate Response
        :param src: [batch_size,sent_max_len]
        :param sents_length: [batch_size]
        :param A_res: [batch_size,user_dim]
        :param A_tgt: [batch_size,user_dim]
        :return: predict
        """
        greedy_dec_input = self.greedy_decode(src, sents_length, A_res, A_tgt)

        response_length = torch.tensor([greedy_dec_input.size(0)], dtype=torch.long).to(args.device)
        generate_output = self(src, sents_length, greedy_dec_input, response_length, A_res, A_tgt)  # [batch_size*tgt_max_len,vocab_size]
        predict = generate_output.data.max(1, keepdim=False)[1]
        return predict  # [tgt_max_len]


if __name__ == '__main__':
    model = MyTransformer()
    temp_src = torch.tensor(np.array([[1, 2, 3, 4, 5, 0, 0], [6, 7, 8, 9, 0, 0, 0]]), dtype=torch.long).transpose(0, 1)
    temp_tgt = torch.tensor(np.array([[9, 8, 7, 6, 0, 0], [5, 4, 3, 0, 0, 0]]), dtype=torch.long).transpose(0, 1)
    print(temp_src.size())
    print(temp_tgt.size())

    src_mask = src_tgt_mask(7)
    # src_key_padding_mask = src_tgt_key_padding_mask(torch.LongTensor([5, 4]))

    output = model(temp_src, temp_tgt, None, None, src_mask)
