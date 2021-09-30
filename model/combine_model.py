# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate
@File: combine_model.py
@Author: QI
@Date: 2021/6/7 16:45
@Description: 组合模型
"""

import torch
import torch.nn as nn

from config import args
from utils import Vocab
from model.select_model import AdrSelection

if args.use_tf_api:
    if args.generate_way == 0:
        from model.generate_api_0 import MyTransformer
    elif args.generate_way == 1:
        from model.generate_api_1 import MyTransformer
    elif args.generate_way == 2:
        from model.generate_api_2 import MyTransformer
    elif args.generate_way == 3:
        from model.generate_api_3 import MyTransformer
# else:
#     if args.generate_way == 0:
#         from model.generate_model0 import MyTransformer
#     elif args.generate_way == 1:
#         from model.generate_model1 import MyTransformer
#     elif args.generate_way == 2:
#         from model.generate_model2 import MyTransformer


class ASRGMPC(nn.Module):
    def __init__(self):
        super(ASRGMPC, self).__init__()
        self.adr_select = AdrSelection()
        self.transformer = MyTransformer()

    def forward(self, dig, dig_sent_length, dig_users, responder, join_sents, join_sents_length, response, response_length, target_user):
        select_output, A = self.adr_select(dig, dig_sent_length, dig_users, responder)

        # 获取一个batch中当前说话人向量
        A_res = torch.zeros([args.batch_size, args.user_dim], dtype=torch.float32, requires_grad=False).to(args.device)  # [batch_size,user_dim]
        for i in range(args.batch_size):
            A_res[i, :] = A[i, responder[i], :]

        A_tgt = torch.zeros([args.batch_size, args.user_dim], dtype=torch.float32, requires_grad=False).to(args.device)  # [batch_size,user_dim]
        for i in range(args.batch_size):
            A_tgt[i, :] = A[i, target_user[i], :]

        if args.use_tf_api:
            generate_output = self.transformer(join_sents, join_sents_length, response, response_length, A_res, A_tgt)
        else:
            generate_output = self.transformer(join_sents, response, A_res, A_tgt)

        return select_output, generate_output, A_res, A_tgt


if __name__ == '__main__':
    model = ASRGMPC()

    for name in model.state_dict():
        print(name)
