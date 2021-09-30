# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate 
@File: dingding.py
@Author: QI
@Date: 2021/7/4 21:22 
@Description: None
"""

import torch
from tqdm import tqdm

from config import args
from dataset import my_dataloader
from utils import Vocab, load_pickle, load_data_talk
from model.combine_model import ASRGMPC


@torch.no_grad()
def interact(model):
    model.eval()
    args.mode = "test"
    args.shuffle = False
    args.batch_size = 1

    vocab = load_pickle(r"../data/vocab_" + str(int(args.cutoff / 10000)) + "w.pkl")

    load_data_talk("./dialogue.txt", vocab)
    talk_dataloader = my_dataloader(mode=args.mode, talk=True)

    model.load_state_dict(torch.load("../save/model.pt", map_location=args.device))

    for idx, (dig, dig_sent_length, dig_name_list, dig_users, responder, target_user, join_sents, join_sents_length, response, response_length) in enumerate(talk_dataloader):
        select_output, generate_output, a_res, a_tgt = model(dig, dig_sent_length, dig_users, responder, join_sents, join_sents_length, response[:, :-1],
                                                             response_length,
                                                             target_user)
        # select_output:[batch_size,max_roles-1], generate_output:[batch_size * tgt_max_len, vocab_size]
        # addressee预测
        select_output = torch.softmax(select_output, dim=-1)  # [batch_size,max_roles-1]
        logit, index = torch.max(select_output, dim=-1)  # [batch_size]

        # 获取概率值
        logit = logit.tolist()
        logit = round(logit[0], 2)

        # 生成回复
        greedy_dec_outputs = model.transformer.greedy_decoder(join_sents, join_sents_length, a_res, a_tgt, vocab.SOS)
        generate_response = vocab.digits2words(greedy_dec_outputs.squeeze(-1))

        target_addressee = dig_name_list[0][index.item()]
        target_message = " ".join(generate_response)

        return target_addressee, target_message, logit
