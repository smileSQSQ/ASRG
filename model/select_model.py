# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate
@File: select_model.py
@Author: QI
@Date: 2021/6/7 14:20
@Description: 回复对象选择模型
"""

import torch
import torch.nn as nn

from config import args
from model.component import UpdateSpeaker, UpdateAddressee, UpdateOthers, LuongAttention, SelfAttention
from utils import Vocab, load_pickle


class UtteranceEncoder(nn.Module):
    def __init__(self):
        super(UtteranceEncoder, self).__init__()
        vocab = load_pickle(r"../data/vocab_" + str(int(args.cutoff / 10000)) + "w.pkl")
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim).from_pretrained(torch.FloatTensor(vocab.pretrained_embeddings), freeze=False)
        self.gru = nn.GRU(args.embedding_dim, args.utter_hidden, args.num_layers, batch_first=True, dropout=args.dropout, bidirectional=True)

    def forward(self, dig, dig_sent_length):
        """
        UtteranceEncoder
        @param dig: [batch,window,seq_len]
        @param dig_sent_length: [batch,window]
        @return: encoder_hiddens
        """
        encoder_hiddens = torch.zeros([args.batch_size, args.context_window, args.utter_hidden * 2], dtype=torch.float32, requires_grad=False).to(args.device)

        for T in range(args.context_window):
            inpt = dig[:, T, :]  # [batch_size,seq_len]
            inpt_length = dig_sent_length[:, T]  # [batch_size]

            embedded = self.embedding(inpt)  # [batch_size,seq_len,embedding_dim]
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, inpt_length, batch_first=True, enforce_sorted=False)

            h0 = torch.zeros(args.num_layers * 2, args.batch_size, args.utter_hidden).to(args.device)
            encoder_output, encoder_hidden = self.gru(embedded, h0)
            # encoder_output: [batch_size,seq_len,utter_hidden*2], encoder_hidden: [num_layers*2,batch_size,utter_hidden]

            fw = encoder_hidden[-2, :, :]  # [batch_size,utter_hidden]
            bw = encoder_hidden[-1, :, :]
            encoder_hidden = torch.cat([fw, bw], dim=-1)  # [batch_size,utter_hidden*2]
            encoder_hiddens[:, T, :] = encoder_hidden

        return encoder_hiddens


class InteractionLayer(nn.Module):
    def __init__(self):
        super(InteractionLayer, self).__init__()
        self.update_speaker = UpdateSpeaker()
        self.update_addressee = UpdateAddressee()
        self.update_others = UpdateOthers()

    def forward(self, encoder_hiddens, dig_users):
        """
        GRUS, GRUA, GRUO
        @param encoder_hiddens:[batch_size,window,utter_hidden*2]
        @param dig_users:[batch_size,window,2]
        @return:A
        """
        A = torch.zeros([args.batch_size, args.max_roles, args.user_dim], dtype=torch.float32, requires_grad=False).to(args.device)

        for T in range(args.context_window):
            encoder_hidden = encoder_hiddens[:, T, :]  # [batch_size,utter_hidden*2]

            spk_id = dig_users[:, T, 0]  # [batch_size]
            adr_id = dig_users[:, T, 1]
            others_id = torch.zeros([args.batch_size, args.max_roles - 2], dtype=torch.long, requires_grad=False).to(args.device)
            for i in range(args.batch_size):
                temp = torch.tensor([idx for idx in range(args.max_roles) if idx != spk_id[i] and idx != adr_id[i]], dtype=torch.long, requires_grad=False).to(args.device)
                others_id[i, :] = temp  # [batch_size,max_roles-2]

            # T时刻，获取一个batch中该句子的spk_vectors,adr_vectors,others_vectors
            spk_vectors = torch.zeros([args.batch_size, args.user_dim], dtype=torch.float32, requires_grad=False).to(args.device)  # [batch_size,user_dim]
            for i in range(args.batch_size):
                spk_vectors[i, :] = A[i, spk_id[i], :]
            adr_vectors = torch.zeros([args.batch_size, args.user_dim], dtype=torch.float32, requires_grad=False).to(args.device)  # [batch_size,user_dim]
            for i in range(args.batch_size):
                adr_vectors[i, :] = A[i, adr_id[i], :]
            others_vectors = torch.zeros([args.batch_size, args.max_roles - 2, args.user_dim], dtype=torch.float32, requires_grad=False).to(args.device)
            # others_vectors: [batch_size,max_roles-2,user_dim]
            for i in range(args.batch_size):
                for j in range(args.max_roles - 2):
                    others_vectors[i, j, :] = A[i, others_id[i, j], :]

            # update speaker,update A
            new_spk_vectors = self.update_speaker(encoder_hidden, spk_vectors, adr_vectors)
            for i in range(args.batch_size):
                A[i, spk_id[i], :] = new_spk_vectors[i, :]

            # update addressee,update A
            new_adr_vectors = self.update_addressee(encoder_hidden, adr_vectors, spk_vectors)
            for i in range(args.batch_size):
                A[i, adr_id[i], :] = new_adr_vectors[i, :]

            # update others,update A
            new_others_vectors = self.update_others(encoder_hidden, others_vectors)
            for i in range(args.batch_size):
                for j in range(args.max_roles - 2):
                    A[i, others_id[i, j], :] = new_others_vectors[i, j, :]

        return A  # [batch_size, max_roles, user_dim]


class AdrSelection(nn.Module):
    def __init__(self):
        super(AdrSelection, self).__init__()
        self.utter_encoder = UtteranceEncoder()
        self.interaction_layer = InteractionLayer()

        if args.select_way == 0:
            # 方案0（论文源方案）: A by max-pooling
            self.max_pooling1 = nn.MaxPool1d(8, 8)
            self.max_pooling2 = nn.MaxPool1d(3, 1)
            self.fc1 = nn.Linear(args.user_dim * 2, args.user_dim)
            self.fc2 = nn.Linear(args.max_roles - 1, args.max_roles - 1)
        elif args.select_way == 1:
            # 方案一: encoder_hiddens by conv
            self.conv1 = nn.Conv1d(in_channels=args.context_window, out_channels=args.context_window, kernel_size=2, stride=2)
            self.fc1 = nn.Linear(args.context_window * args.utter_hidden, args.user_dim)
            self.fc2 = nn.Linear(args.user_dim * 2, args.user_dim)
            self.fc3 = nn.Linear(args.max_roles - 1, args.max_roles - 1)
        elif args.select_way == 2:
            # 方案二: encoder_hiddens by gru
            self.gru = nn.GRU((args.utter_hidden * 2), args.user_dim, batch_first=True)
            self.fc1 = nn.Linear(args.user_dim * 2, args.user_dim)
            self.fc2 = nn.Linear(args.max_roles - 1, args.max_roles - 1)
        elif args.select_way == 3:
            # 方案三: encoder_hiddens by attn and a_p
            self.attn = LuongAttention(args.attn_method, args.utter_hidden * 2)
            self.conv1 = nn.Conv1d(in_channels=args.max_roles - 1, out_channels=args.max_roles - 1, kernel_size=2, stride=2)
            self.fc1 = nn.Linear((args.max_roles - 1) ** 2, args.max_roles - 1)
        elif args.select_way == 4:
            # 方案四: encoder_hiddens by self-attn
            self.self_attn = SelfAttention(args.utter_hidden * 2)
            self.fc1 = nn.Linear(args.user_dim * 2, args.user_dim)
            self.fc2 = nn.Linear(args.max_roles - 1, args.max_roles - 1)

    def forward(self, dig, dig_sent_length, dig_users, responder):
        """
        AdrSelection
        @param dig: [batch_size,context_window,seq_len]
        @param dig_sent_length: [batch_size,context_window]
        @param dig_users: [batch_size,context_window,2]
        @param responder: [batch_size]
        @return: output, A
        """
        encoder_hiddens = self.utter_encoder(dig, dig_sent_length)  # [batch_size,context_window,utter_hidden*2]
        A = self.interaction_layer(encoder_hiddens, dig_users)  # [batch_size,max_roles,user_dim]

        # 获取一个batch中当前说话人向量
        A_res = torch.zeros([args.batch_size, args.user_dim], dtype=torch.float32, requires_grad=False).to(args.device)  # [batch_size,user_dim]
        for i in range(args.batch_size):
            A_res[i, :] = A[i, responder[i], :]
        # 获取一个batch中所有可能回复的对象id
        p_tgt_ids = torch.zeros([args.batch_size, args.max_roles - 1], dtype=torch.long, requires_grad=False).to(args.device)  # [batch_size,max_roles-1]
        for i in range(args.batch_size):
            temp = torch.tensor([j for j in range(args.max_roles) if j != responder[i]], dtype=torch.long, requires_grad=False).to(args.device)
            p_tgt_ids[i, :] = temp
        # 获取所有可能回复对象的向量
        p_tgt_vectors = torch.zeros([args.batch_size, args.max_roles - 1, args.user_dim], dtype=torch.float32, requires_grad=False).to(args.device)
        # p_tgt_vectors: [batch_size,max_roles-1,user_dim]
        for i in range(args.batch_size):
            for j in range(args.max_roles - 1):
                p_tgt_vectors[i, j, :] = A[i, p_tgt_ids[i, j], :]

        output = None
        if args.select_way == 0:
            # # 方案0（论文源方案）: A by max-pooling
            temp = self.max_pooling1(A)
            temp = temp.permute(0, 2, 1)
            temp = self.max_pooling2(temp)
            context_vector = temp.reshape(temp.size(0), -1)  # [batch_size,user_dim]
            concated = torch.cat([A_res, context_vector], dim=-1)  # [batch_size,user_dim*2]
            concated = torch.tanh(self.fc1(concated))  # [batch_size,user_dim]
            concated = concated.unsqueeze(dim=-1)  # [batch_size,user_dim,1]
            output = torch.bmm(p_tgt_vectors, concated)  # [batch_size,roles-1,1]
            output = output.view(args.batch_size, -1)  # [batch_size,roles-1]
            output = self.fc2(output)  # [batch_size,max_roles-1]
        elif args.select_way == 1:
            # 方案一: encoder_hiddens by conv
            encoder_hiddens = self.conv1(encoder_hiddens)  # [batch_size,window,utter_hidden]
            encoder_hiddens = encoder_hiddens.view(args.batch_size, -1)  # [batch_size,window*utter_hidden]
            encoder_hiddens = torch.tanh(self.fc1(encoder_hiddens))  # [batch_size,user_dim]
            concated = torch.cat([A_res, encoder_hiddens], dim=-1)  # [batch_size,user_dim*2]
            concated = torch.tanh(self.fc2(concated))  # [batch_size,user_dim]
            concated = concated.unsqueeze(dim=-1)  # [batch_size,user_dim,1]
            output = torch.bmm(p_tgt_vectors, concated)  # [batch_size,roles-1,1]
            output = output.view(args.batch_size, -1)  # [batch_size,roles-1]
            output = self.fc3(output)  # [batch_size,max_roles-1]
        elif args.select_way == 2:
            # 方案二: encoder_hiddens by gru
            h0 = torch.zeros(1, args.batch_size, args.user_dim).to(args.device)
            _, context_hidden = self.gru(encoder_hiddens, h0)  # [1,batch_size,user_dim]
            concated = torch.cat([A_res, context_hidden.squeeze(dim=0)], dim=-1)  # [batch_size,user_dim*2]
            concated = torch.tanh(self.fc1(concated))  # [batch_size,user_dim]
            concated = concated.unsqueeze(dim=-1)  # [batch_size,user_dim,1]
            output = torch.bmm(p_tgt_vectors, concated)  # [batch_size,roles-1,1]
            output = output.view(args.batch_size, -1)  # [batch_size,roles-1]
            output = self.fc2(output)  # [batch_size,max_roles-1]
        elif args.select_way == 3:
            # 方案三: encoder_hiddens by attn and a_p
            concated = torch.zeros([args.batch_size, args.max_roles - 1, args.user_dim * 2], dtype=torch.float32, requires_grad=False).to(args.device)
            for i in range(args.max_roles - 1):
                a_p = p_tgt_vectors[:, i, :]  # [batch_size,user_dim]
                # attn_weights = self.attn(a_p.unsqueeze(1).repeat(1, args.context_window, 1), encoder_hiddens)  # [batch_size,1,window]
                attn_weights = self.attn(a_p.unsqueeze(1), encoder_hiddens)  # [batch_size,1,window]
                context_vector = attn_weights.bmm(encoder_hiddens)  # [batch_size,1,user_dim]
                context_vector = context_vector.squeeze(1)  # [batch_size,user_dim]
                cat = torch.cat([A_res, context_vector], dim=-1)  # [batch_size,user_dim*2]
                concated[:, i, :] = cat

            concated = self.conv1(concated)  # [batch_size,max_roles-1,user_dim]
            output = torch.bmm(concated, p_tgt_vectors.permute(0, 2, 1))  # [batch_size,max_roles-1,max_roles-1]
            output = torch.tanh(self.fc1(output.view(args.batch_size, -1)))  # [batch_size,max_roles-1]
            output = torch.tanh(output)  # [batch_size,max_roles-1]
        elif args.select_way == 4:
            # 方案四: encoder_hiddens by self-attn
            context_vector, _ = self.self_attn(encoder_hiddens)  # [batch_size,user_dim]
            concated = torch.cat([A_res, context_vector], dim=-1)  # [batch_size,user_dim*2]
            concated = torch.tanh(self.fc1(concated))  # [batch_size,user_dim]
            concated = concated.unsqueeze(dim=-1)  # [batch_size,user_dim,1]
            output = torch.bmm(p_tgt_vectors, concated)  # [batch_size,roles-1,1]
            output = output.view(args.batch_size, -1)  # [batch_size,roles-1]
            output = self.fc2(output)  # [batch_size,max_roles-1]

        return output, A  # [batch,max_roles-1]


if __name__ == '__main__':
    model = AdrSelection()
    print(model)

    for name in model.state_dict():
        print(name)
