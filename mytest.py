import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io

# A = torch.randint(1, 10, [3, 5, 10], dtype=torch.int64)  # [batch,roles,dim]
# print(A)
# responder = [1, 4, 0]  # [batch]
# p_tgt = torch.zeros([3, 4], dtype=torch.int64)  # [batch,roles-1]
# for i in range(3):
#     temp = torch.tensor([j for j in range(5) if j is not responder[i]])
#     p_tgt[i, :] = temp
# print(p_tgt)
#
# p_tgt_vectors = torch.LongTensor(torch.zeros([3, 4, 10], dtype=torch.int64))  # [batch_size,max_roles-1,user_dim]
# for i in range(3):
#     for j in range(4):
#         p_tgt_vectors[i, j, :] = A[i, p_tgt[i, j], :]
# print(p_tgt_vectors)

# spk_id = [1, 4, 0]
# adr_id = [0, 2, 3]
# others_id = torch.zeros([3, 3],dtype=torch.int64)
# for i in range(3):
#     temp = torch.tensor([id for id in range(5) if id is not spk_id[i] and id is not adr_id[i]])
#     others_id[i, :] = temp
# print(others_id)
# print(others_id[0,0])
#
# spk_vectors = torch.zeros([3, 3, 10])
# for i in range(3):
#     for j in range(3):
#         spk_vectors[i, j, :] = A[i, others_id[i, j], :]
# print(spk_vectors)

# spk_vectors = torch.zeros([3, 10])
# for i in range(3):
#     spk_vectors[i, :] = A[i, spk_id[i], :]
# print(spk_vectors)

# r_t = torch.ones([3, 10]) + torch.ones([3, 10]) + torch.ones([3, 10])
# print(r_t.size())


# spk_vectors = torch.ones([3, 10]) * 3
# for i in range(3):
#     A[i, spk_id[i], :] = spk_vectors[i, :]
# print(A)

# conv1 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=2, stride=2)
# input = torch.randn(64, 5, 1024)
# out = conv1(input)
# print(out.size())


# x = torch.ones([3, 1, 10])
# # out:[3,5,10]
# print(x.size())
# print(x)
# x = x.repeat(1, 5, 1)
# print(x.size())
# print(x)


# hs = torch.zeros([3, 5, 10])
# h = torch.ones([3, 10])
# # h = h.unsqueeze(dim=1)
# hs[:, 0, :] = h
# print(hs)


# oupt = np.array([[0.1, 0.2, 0.3, 0.1, 0.2],
#                  [0.5, 0.1, 0.1, 0.1, 0.2],
#                  [0.3, 0.4, 0.1, 0.1, 0.1]])
# oupt = torch.tensor(oupt)
# oupt = torch.softmax(oupt, dim=-1)
# print(oupt.size())
# print(oupt)
#
# value, index = torch.max(oupt, dim=-1)
# print(value.size())
# print(value)
# print(index.size())
# print(index)


# a = torch.tensor(np.array([1, 2, 3]))
# b = torch.tensor(np.array([4, 2, 5]))
# print((a == b).sum())


# A_res = torch.tensor(torch.randint(0, 10, [3, 10]))
# print(A_res.size())
# A_res = A_res.unsqueeze(1)
# print(A_res.size())
# A_res = A_res.repeat(1, 5, 1)
# print(A_res.size())

# a = torch.ones([3, 10])
# b = torch.ones([3, 10])
# c = a + b
# print(c)


# def load_vectors(fname="./data/wiki_300d.vec"):
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     count = 1
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         print(tokens)
#         data[tokens[0]] = map(float, tokens[1:])
#         # if count == 1:
#         #     print(data[tokens[0]])
#         #     count += 1
#     return data
#
#
# load_vectors("./data/wiki_300d.vec")


# sent = "I love you".lower().split()
# print(sent)


# FloatTensor containing pretrained weights
# weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
# embedding = nn.Embedding(2, 3).from_pretrained(weight)
# # Get embeddings for index 1
# input = torch.LongTensor([1])
# print(embedding(input))


# class Attn(nn.Module):
#     ''' Luong attention '''
#
#     def __init__(self, method, hidden_size):
#         super(Attn, self).__init__()
#         self.method = method
#         if self.method not in ['dot', 'general', 'concat']:
#             raise ValueError(self.method, "is not an appropriate attention method.")
#         self.hidden_size = hidden_size
#         if self.method == 'general':
#             self.attn = nn.Linear(self.hidden_size, hidden_size)
#         elif self.method == 'concat':
#             self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
#             self.v = nn.Parameter(torch.FloatTensor(hidden_size))
#
#     def dot_score(self, hidden, encoder_output):
#
#         return torch.sum(hidden * encoder_output, dim=2)
#
#     def general_score(self, hidden, encoder_output):
#         energy = self.attn(encoder_output)
#         return torch.sum(hidden * energy, dim=2)
#
#     def concat_score(self, hidden, encoder_output):
#         energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
#         return torch.sum(self.v * energy, dim=2)
#
#     def forward(self, hidden, encoder_outputs):
#         # Calculate the attention weights (energies) based on the given method
#         if self.method == 'general':
#             attn_energies = self.general_score(hidden, encoder_outputs)
#         elif self.method == 'concat':
#             attn_energies = self.concat_score(hidden, encoder_outputs)
#         elif self.method == 'dot':
#             attn_energies = self.dot_score(hidden, encoder_outputs)
#
#         print(attn_energies.size())  # [2,4]
#
#         # Return the softmax normalized probability scores (with added dimension)
#         return F.softmax(attn_energies, dim=1).unsqueeze(1)


# attn = Attn(method='dot', hidden_size=3)
#
# hidden = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)  # [2,3]
# hidden = hidden.unsqueeze(1).repeat(1, 4, 1)  # [2,4,3]
# encoder_output = torch.randn([2, 4, 3], dtype=torch.float32)  # [2,4,3]
# attn_weights = attn(hidden, encoder_output)  # [2,1,4]
# print(attn_weights.size())
# context_vector = attn_weights.bmm(encoder_output)
# print(context_vector.size())


# a=torch.tensor([[1, 2, 3], [4, 5, 6]],dtype=torch.float32)
# b=torch.tensor([[1, 2, 3], [4, 5, 6]],dtype=torch.float32)
# c=torch.matmul()


# a = np.array([
#     [[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7]],
#     [[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7]]
# ])
# b = np.array([
#     [[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7]],
#     [[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7]]
# ])
# a = torch.tensor(a, dtype=torch.float32)
# b = torch.tensor(b, dtype=torch.float32)
# print(a.size())
# print(b.size())
# c = a * b
# print(c.size())
# print(c)
# c = torch.sum(c, dim=2)
# print(c.size())
# print(c)

# a = torch.tensor([[1, 2, 3], [2, 3, 4]])
# b = torch.tensor([[3, 4, 5], [4, 5, 6]])
# c = torch.mm(a, b)
# print(c.size())
# print(c)

#
# class SelfAttention(nn.Module):
#     def __init__(self, hidden_dim):
#         super(SelfAttention, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.projection = nn.Sequential(
#             nn.Linear(hidden_dim, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, encoder_outputs):
#         # encoder_outputs: [batch_size,window,utter_hidden*2]
#         energy = self.projection(encoder_outputs)  # [batch_size,window,1]
#         weights = F.softmax(energy.squeeze(-1), dim=1)  # [batch_size,window]
#
#         temp = encoder_outputs * weights.unsqueeze(-1)  # [batch_size,window,utter_hidden*2]
#         outputs = temp.sum(dim=1)
#         context_vector = outputs  # [batch_size,utter_hidden*2]
#         return context_vector, weights
#
#
# sattn = SelfAttention(512)
# encoder_outputs = torch.randn([32, 10, 512])
# context_vector, weights = sattn(encoder_outputs)

# import os
# print(os.listdir("./data"))


# from utils import Vocab, load_pickle
#
# vocab = load_pickle(r"./data/vocab.pkl")
# print(type(vocab.pretrained_embeddings))
# t = torch.FloatTensor(vocab.pretrained_embeddings)
#
# print(t.size())


# a = torch.zeros([2, 4])
# b = torch.ones([2, 3])
# c = a + b
# print(c)

# def src_key_padding_mask(real_length):
#     mask = torch.zeros([2, 7], dtype=torch.long)
#     for i in range(2):
#         no_padding = torch.zeros([1, real_length[i].item()], dtype=torch.long)
#         padding = torch.ones([1, 7 - real_length[i].item()], dtype=torch.long)
#         mask[i, :] = torch.cat([no_padding, padding], dim=-1)
#     mask = mask.bool()
#     return mask
#
#
# real_length = torch.LongTensor([3, 7])
# mask = src_key_padding_mask(real_length)
# print(mask)


# output = torch.randn([10, 32, 512])
# fc = nn.Linear(512, 128)
# output = fc(output)
# print(output.size())


# from model.combine_model import ASRGMPC
# from torch.utils.tensorboard import SummaryWriter
#
# writer = SummaryWriter("./save/")
#
# model = ASRGMPC()
# writer.add_graph(model)
#
#
# writer.close()


# from config import args
#
# global_step = 1000
#
# if global_step % 500 == 0:
#     args.lr = args.lr * args.decay_rate ** (global_step / args.decay_steps)

# import warnings
# from nlgeval import compute_metrics
#
# warnings.filterwarnings('ignore')
#
# metrics_dict = compute_metrics(hypothesis='examples/hyp.txt', references=['examples/ref1.txt', "examples/ref2.txt"])
# print(metrics_dict)


# print(memory.size())  # [64,80,512]
# self.maxpool1 = nn.MaxPool1d(8, 8)
# output = maxpool1(memory)
# print(output.size())  # [64,80,64]
# output = output.permute(0, 2, 1)  # [64,64,80]
# self.maxpool2 = nn.MaxPool1d(10, 10)
# output = maxpool2(output)
# print(output.size())  # [64,64,8]
# output = output.reshape(output.size(0), -1)
# print(output.size())

# A = torch.randn([64, 10, 512], dtype=torch.float32)
#
# max_pooling1 = nn.MaxPool1d(8, 8)
# max_pooling2 = nn.MaxPool1d(3, 1)
#
# temp = max_pooling1(A)
# temp = temp.permute(0, 2, 1)
# temp = max_pooling2(temp)
# context_vector = temp.reshape(temp.size(0), -1)
# print(context_vector.size())


# a = np.array([
#     [[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7]],
#     [[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7]]
# ])
# b = np.array([
#     [[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7]],
#     [[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7]]
# ])
# a = torch.tensor(a, dtype=torch.float32)
# b = torch.tensor(b, dtype=torch.float32)
# print(a.size())
# print(b.size())
# b = b.add(a)
# print(b.size())
# print(b)

# a = 3
# b = a
# print(b)
# a = 4
# print(b)

# a = torch.ones([10, 32, 50], dtype=torch.float)
# b = torch.ones([1, 32, 50], dtype=torch.float)
# c = torch.cat([b, a], dim=0)
# print(c.size())

# a = torch.ones([4, 8])
# a[:, 1] = torch.zeros([4])
# print(a)

# b = np.array([[1, 2, 3],
#               [2, 3, 1],
#               [3, 4, 5],
#               [9, 5, 7]])
# b = torch.tensor(b, dtype=torch.float32)
# print(b.size())
# c = b.data.max(1, keepdim=True)[1]
# print(c)


# dec_inputs = torch.zeros([10, 1], dtype=torch.long)
# print(dec_inputs.size())
# a = torch.tensor([dec_inputs.size(0)], dtype=torch.long)
# print(a)

# def src_tgt_mask(sz):
#     # 下三角矩阵
#     # sz为padding之后的句子长度, 标量-sents_max_len/target_max_len
#     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask
#
#
# mask = src_tgt_mask(10)
# print(mask)


# def get_attn_pad_mask(seq_q, seq_k):
#     '''
#     seq_q: [batch_size, seq_len]
#     seq_k: [batch_size, seq_len]
#     seq_len could be src_len or it could be tgt_len
#     seq_len in seq_q and seq_len in seq_k maybe not equal
#     '''
#     batch_size, len_q = seq_q.size()
#     batch_size, len_k = seq_k.size()
#     # eq(zero) is PAD token
#     pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
#     return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
#
#
# seq_q = torch.tensor([[1, 2, 3, 0, 0, 0], [3, 4, 5, 6, 0, 0]], dtype=torch.long)
# seq_k = torch.tensor([[1, 2, 3, 4, 0, 0], [3, 4, 0, 0, 0, 0]], dtype=torch.long)
# pad_attn_mask = get_attn_pad_mask(seq_q, seq_k)
# print(pad_attn_mask)

# def get_attn_subsequence_mask(seq):
#     '''
#     seq: [batch_size, tgt_len]
#     '''
#     attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
#     subsequence_mask = torch.triu(torch.ones([seq.size(0), seq.size(1), seq.size(1)]), diagonal=1)  # Upper triangular matrix
#     # subsequence_mask = torch.from_numpy(subsequence_mask).float()
#     return subsequence_mask  # [batch_size, tgt_len, tgt_len]
#
#
# seq = torch.tensor([[1, 2, 3, 0, 0, 0], [3, 4, 5, 6, 0, 0]], dtype=torch.long)
# subsequence_mask = get_attn_subsequence_mask(seq)
# print(subsequence_mask)


# def src_tgt_mask(sz):
#     # 下三角矩阵
#     # sz为padding之后的句子长度, 标量-sents_max_len/target_max_len
#     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask  # [sz,sz]
#
#
# mask = src_tgt_mask(10)
# print(mask)


# enc_inputs = torch.tensor([[1, 2, 3, 0, 0, 0], [3, 4, 5, 6, 0, 0]], dtype=torch.long)
#
# print(enc_inputs[0].view(1, -1))


# class MyTest():
#     def f(self, src):
#         src = src.transpose(0, 1)
#         print(src.size())
#
#     def decoder(self, src):
#         print(src.size())
#
#     def gene(self, src):
#         src = src.transpose(0, 1)
#         self.decoder(src)
#         print(src.size())
#
#
# my_test = MyTest()
# src = torch.tensor([[1, 2, 3, 0, 0, 0], [3, 4, 5, 6, 0, 0]], dtype=torch.long)
# my_test.f(src)
# my_test.gene(src)

#
# class TestDeliver:
#     def fun(self, ans):
#         ans += 1
#         print(ans)
#
#
# c = TestDeliver()
# ans = 1
# c.fun(ans)
# print(ans)


# # 可视化实验
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import manifold
#
# #  准备数据
# s1 = np.random.normal(0, 1, (20, 300))
# s2 = np.random.randn(20, 300)
# # u = np.random.randn(1, 300).repeat(2, 0)
# # u = np.array([[2.2, 3.3, 4.8, 2, 1.5], [1.2, 2.3, 4, 2, 2.5]])
# u = np.array([[2.2, 3.3, 4.8, 2, 1.5]])
# print(u.shape)
# # 降维
# tsne = manifold.TSNE(n_components=2, init='pca')
#
# s1_tsne = tsne.fit_transform(s1)
# s2_tsne = tsne.fit_transform(s2)
# u_tsne = tsne.fit_transform(u)
# print("s1: Org data dimension is {}.Embedded data dimension is {}".format(s1.shape[-1], s1_tsne.shape[-1]))
# print("s2: Org data dimension is {}.Embedded data dimension is {}".format(s2.shape[-1], s2_tsne.shape[-1]))
# print("#####################################")
# print("u:{}".format(u_tsne.shape))
# print(u_tsne)
# # 归一化
# s1_min, s1_max = s1_tsne.min(0), s1_tsne.max(0)
# s1_norm = (s1_tsne - s1_min) / (s1_max - s1_min)  # 归一化
# s2_min, s2_max = s2_tsne.min(0), s2_tsne.max(0)
# s2_norm = (s2_tsne - s2_min) / (s2_max - s2_min)  # 归一化
# u_min, u_max = u_tsne.min(0), u_tsne.max(0)
# u_norm = (u_tsne - u_min) / (u_max - u_min)  # 归一化
# # 嵌入空间可视化
# plt.title("Correlation between generated content and virtual user")
# plt.scatter(s1_norm[:, 0], s1_norm[:, 1], marker='o', label="My Model")
# plt.scatter(s2_norm[:, 0], s2_norm[:, 1], marker='o', label="Original Transformer")
# plt.scatter(u_norm[0, 0], u_norm[0, 1], marker='^', label="Addressee")
# plt.legend(loc='upper right')
# plt.show()
#
# temp = np.array([1, 2, 3])
# print(temp[:-1])


# print(int(4/2))


import numpy as np


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn import manifold
#
#
# def plot_correlation_3d(s1, s2, u):
#     """
#     绘制Addressee与生成内容的相关性3维散点图
#     :param s1: Our Focus Transformer Model生成的句子, [len,dim]
#     :param s2: Original Transformer生成的句子, [len,dim]
#     :param u: Addressee表示向量, 从A中取得, [user_dim]
#     :return: 3维散点图
#     """
#     # 降维
#     tsne = manifold.TSNE(n_components=3, init='pca')
#     s1_tsne = tsne.fit_transform(s1)
#     s2_tsne = tsne.fit_transform(s2)
#     u_tsne = tsne.fit_transform(u)
#     # 归一化
#     s1_min, s1_max = s1_tsne.min(0), s1_tsne.max(0)
#     s1_norm = (s1_tsne - s1_min) / (s1_max - s1_min)
#     s2_min, s2_max = s2_tsne.min(0), s2_tsne.max(0)
#     s2_norm = (s2_tsne - s2_min) / (s2_max - s2_min)
#     u_min, u_max = u_tsne.min(0), u_tsne.max(0)
#     u_norm = (u_tsne - u_min) / (u_max - u_min)
#     # 嵌入空间可视化
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.set_title("Correlation Between Generated Content And Addressee")
#
#     ax.scatter(s1_norm[:, 0], s1_norm[:, 1], s1_norm[:, 2], marker='o', label="Our Focus Transformer Model")
#     ax.scatter(s2_norm[:, 0], s2_norm[:, 1], s2_norm[:, 2], marker='o', label="Original Transformer")
#     ax.scatter(u_norm[0, 0], u_norm[0, 1], u_norm[0, 2], marker='^', label="Addressee")
#     ax.legend(loc='best')
#     plt.show()
#
#
# #  准备数据
# s1 = np.random.normal(0, 1, (20, 300))
# s2 = np.random.randn(20, 300)
# u = np.random.randn(5, 300).repeat(2, 0)
#
# # plot_correlation_3d(s1, s2, u)
#
# # a_tgt = torch.zeros([1, 512], dtype=torch.float32)
# # print(a_tgt.squeeze(0))
#
# tgt = np.array([])
# for i in range(10):
#     a_tgt = torch.zeros([1, 512], dtype=torch.float32)
#     a_tgt = a_tgt.squeeze(0)
#     np.append(tgt, a_tgt, 0)
# print(tgt.shape)

def IdName(members):
    '''
    建立openId和FullName之间的映射
    :param members: [{'FullName': '', 'groupNick': '', 'role': '', 'openId': },{'FullName': '', 'groupNick': '', 'role': '', 'openId': }......]
    :return: dict
    '''
    openid2name = {}
    name2openid = {}
    for item in members:
        name = item["FullName"]
        openid = str(item["openId"])
        openid2name[openid] = name
        name2openid[name] = openid
    return openid2name, name2openid


members = [{'FullName': '宋奇', 'groupNick': '', 'role': '群主', 'openId': 1130149588}, {'FullName': '虚拟用户', 'groupNick': '', 'role': '群主', 'openId': 2376198925}]
openid2name, name2openid = IdName(members)

list = [['大家好！我是 虚拟用户 机器人，很高兴为你们服务。', 1625404718454, 2376198925, '虚拟用户'],
        ['试下', 1625404796067, 1130149588, '宋奇'],
        ['我是虚拟用户，测试1！@10 ', 1625405144826, 2376198925, '虚拟用户'],
        ['19821810536你好！\n@1130149588 ', 1625405226049, 2376198925, '虚拟用户'],
        ['试下', 1625407446493, 1130149588, '宋奇'],
        ['text', 1625455061632, 1130149588, '宋奇'],
        ['text', 1625456028342, 1130149588, '宋奇'],
        ['text@宋奇', 1625456394559, 1130149588, '宋奇']]

last_messages = list[-5:]
f = open("./dialogue.txt", "w", encoding="utf-8")
for item in last_messages:
    sender = item[3]
    message = item[0]
    if message.find('@') == -1:
        utterance = message.replace('\n', '').replace('\r', '')
        addressee = '-'
    else:
        utterance, addressee = message.split('@')
        utterance = utterance.replace('\n', '').replace('\r', '')
        addressee = openid2name.get(addressee, '-')

    f.write(sender + "\t" + addressee + "\t")
    f.write(utterance + "\n")

f.close()
