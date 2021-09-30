# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate 
@File: my_test.py
@Author: QI
@Date: 2021/7/6 9:09 
@Description: None
"""

# from utils import Vocab, load_pickle
# from dingtalk.talk_model import interact
# import time
# from model.combine_model import ASRGMPC
# from config import args
#
# vocab = load_pickle(r"../data/vocab_" + "2w.pkl")
#
# # 调用模型生成
# model = ASRGMPC().to(args.device)
# target_addressee, target_message = interact(model)
# print(target_addressee)
# print(target_message)

# at_mobiles = ['sq-19821810536'.split('-')[1]]
# print(at_mobiles[0])

# from dingtalk.talk_utils import *
#
# dt = Work()
# dt.Login()
# cid_dict = dt.getConversation()
# cid = cid_dict['新手体验群']
# members = dt.getUsers(cid)
# print(dt.formatMembersInfo(members))

import torch

select_output = torch.tensor([[2, 4, 1, 3, 5, 2, 1, 8, 6, 4]], dtype=torch.float)
select_output = torch.softmax(select_output, dim=-1)  # [batch_size,max_roles-1]
logit, index = torch.max(select_output, dim=-1)  # [batch_size]
logit = logit.tolist()
print(logit)
logit = round(logit[0], 2)

print(logit)
print(index)


'''
[22:32]	<Lord_Myth>	-	does ubuntu hate to install over another distro on a dual-bot system with windows ? is that my issue ?
[22:32]	<wldcordeiro>	<goddard>	where is the log ?
[22:32]	<goddard>	<wldcordeiro>	just open the dash and type log
[22:33]	<goddard>	<Lord_Myth>	yes it can
[22:33]	<Lord_Myth>	-	then it is all me , and that is worse
[22:33]	<goddard>	<Lord_Myth>	though it may not work `` hassle free '' depending on some partition schemes
'''
