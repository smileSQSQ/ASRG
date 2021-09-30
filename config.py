# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate
@File: config.py
@Author: QI
@Date: 2021/6/1 14:48
@Description: 配置文件
"""

import torch
import argparse

parser = argparse.ArgumentParser(description='配置信息')
##################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
parser.add_argument('--print_steps', type=int, default=300, help='相关信息打印间隔')
parser.add_argument('--mode', type=str, default="test", help='train/dev/test')
parser.add_argument('--select_way', type=int, default=3, help='方案: 0,1,2,3,4')
parser.add_argument('--generate_way', type=int, default=3, help='方案: 0,1,2,3')
parser.add_argument('--lamda1', type=float, default=1.0, help='选择模型loss1超参数')
parser.add_argument('--lamda2', type=float, default=1.0, help='生成回复loss2超参数')
###################################################################################
parser.add_argument('--device', type=str, default=device, help='cpu/gpu')
parser.add_argument('--gpus', type=int, default=1, help='gpu个数')
parser.add_argument('--cutoff', type=int, default=20000, help='词表中词的个数')
parser.add_argument('--vocab_size', type=int, default=20005)
parser.add_argument('--use_pretrain_tf', type=bool, default=False, help='是否加载预训练的Transformer')
parser.add_argument('--use_pretrain_selection', type=bool, default=False, help='是否加载预训练的选择模型')
parser.add_argument('--select_parameters_fix', type=bool, default=False, help='已训练选择模型是否使用固定参数')
parser.add_argument('--use_tf_api', type=bool, default=True, help='Transformer是否使用pytorch API')
parser.add_argument('--use_pretrained_embeddings', type=bool, default=True, help='是否使用预训练的词向量')
parser.add_argument('--input_max_len', type=int, default=20)
parser.add_argument('--target_max_len', type=int, default=22)
parser.add_argument('--max_roles', type=int, default=10, help='群聊参与者最大值')
parser.add_argument('--context_window', type=int, default=5, help='上下文窗口大小')
parser.add_argument('--shuffle', type=bool, default=True, help='数据集是否打乱')
parser.add_argument('--utter_hidden', type=int, default=256, help='句编码器隐层维度')
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--attn_method', type=str, default="dot", help='dot/general/concat')
parser.add_argument('--user_dim', type=int, default=512)
# transformer
parser.add_argument('--sents_max_len', type=int, default=80)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--t_dropout', type=float, default=0.1)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--d_k', type=int, default=64, help='d_k/d_q')
parser.add_argument('--d_v', type=int, default=64)
parser.add_argument('--d_ff', type=int, default=2048)
#####################################
parser.add_argument('--lr_steps', type=int, default=2000, help='lr衰减')
parser.add_argument('--decay_rate', type=float, default=0.9, help='学习率衰减系数')
parser.add_argument('--decay_steps', type=int, default=2000, help='学习率衰减速度')

args = parser.parse_args()
