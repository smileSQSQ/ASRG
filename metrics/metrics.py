# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate
@File: metrics.py
@Author: QI
@Date: 2021/6/1 14:48
@Description: 评价方法
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from config import args
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']


def print_loss(idx, epo, loss, loss1, loss2, running_loss, running_loss1, running_loss2):
    print(f'\nInstant: epoch{epo + 1}-{idx + 1}, loss={loss.item()} (loss1={loss1.item()}, loss2={loss2.item()})')

    print(f'Average: epoch{epo + 1}-{idx + 1}, running_loss={running_loss / args.print_steps} '
          f'(running_loss1={running_loss1 / args.print_steps},'
          f' running_loss2={running_loss2 / args.print_steps})')
    print("\n")


def cal_acc_batch(select_output, target_user):
    """
    计算每个batch中Addressee预测正确的个数
    :param select_output: [batch_size,max_roles-1]
    :param target_user: [batch_size]
    :return: count
    """
    select_output = torch.softmax(select_output, dim=-1)  # [batch_size,max_roles-1]
    logit, index = torch.max(select_output, dim=-1)  # [batch_size]
    right = (index == target_user).sum().item()
    return index, right


def cal_acc(right, total):
    acc = right / total
    return acc


def cal_perplexity():
    pass


def plot_correlation_2d(s1, s2, u):
    """
    绘制Addressee与生成内容的相关性2维散点图
    :param s1: Our Focus Transformer Model生成的句子, [len,dim]
    :param s2: Original Transformer生成的句子, [len,dim]
    :param u: Addressee表示向量, 从A中取得, [user_dim]
    :return: 2维散点图
    """
    # 降维
    tsne = manifold.TSNE(n_components=2, init='pca')
    s1_tsne = tsne.fit_transform(s1)
    s2_tsne = tsne.fit_transform(s2)
    u_tsne = tsne.fit_transform(u)
    # 归一化
    s1_min, s1_max = s1_tsne.min(0), s1_tsne.max(0)
    s1_norm = (s1_tsne - s1_min) / (s1_max - s1_min)
    s2_min, s2_max = s2_tsne.min(0), s2_tsne.max(0)
    s2_norm = (s2_tsne - s2_min) / (s2_max - s2_min)
    u_min, u_max = u_tsne.min(0), u_tsne.max(0)
    u_norm = (u_tsne - u_min) / (u_max - u_min)
    # 嵌入空间可视化
    plt.title("回复对象相关性")
    plt.scatter(s1_norm[:, 0], s1_norm[:, 1], marker='o', label="聚焦Transformer")
    plt.scatter(s2_norm[:, 0], s2_norm[:, 1], marker='o', label="原始Transformer")
    plt.scatter(u_norm[0, 0], u_norm[0, 1], marker='^', label="回复对象")
    plt.legend(loc='upper right')
    plt.show()


def plot_correlation_3d(s1, s2, u):
    """
    绘制Addressee与生成内容的相关性3维散点图
    :param s1: Our Focus Transformer Model生成的句子, [len,dim]
    :param s2: Original Transformer生成的句子, [len,dim]
    :param u: Addressee表示向量, 从A中取得, [user_dim]
    :return: 3维散点图
    """
    # 降维
    tsne = manifold.TSNE(n_components=3, init='pca')
    s1_tsne = tsne.fit_transform(s1)
    s2_tsne = tsne.fit_transform(s2)
    u_tsne = tsne.fit_transform(u)
    # 归一化
    s1_min, s1_max = s1_tsne.min(0), s1_tsne.max(0)
    s1_norm = (s1_tsne - s1_min) / (s1_max - s1_min)
    s2_min, s2_max = s2_tsne.min(0), s2_tsne.max(0)
    s2_norm = (s2_tsne - s2_min) / (s2_max - s2_min)
    u_min, u_max = u_tsne.min(0), u_tsne.max(0)
    u_norm = (u_tsne - u_min) / (u_max - u_min)
    # 嵌入空间可视化
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title("回复相关性")

    ax.scatter(s1_norm[:, 0], s1_norm[:, 1], s1_norm[:, 2], marker='o', label="聚焦Transformer")
    ax.scatter(s2_norm[:, 0], s2_norm[:, 1], s2_norm[:, 2], marker='o', label="原始Transformer")
    ax.scatter(u_norm[0, 0], u_norm[0, 1], u_norm[0, 2], marker='^', label="回复对象")
    ax.legend(loc='best')
    plt.show()


def plt_hot():
    names = ['10', '20', '30', '40', '50']
    x = range(len(names))
    y1 = [4 / 8, 7 / 15, 10 / 23, 19 / 28, 22 / 33]
    y2 = [6 / 8, 13 / 15, 20 / 23, 18 / 28, 20 / 33]
    yn = [0, 0, 0, 1 / 4, 8 / 9]

    plt.ylim(0, 1)  # 限定纵轴的范围

    plt.plot(x, y1, mec='r', mfc='w', label=u'用户1')
    plt.plot(x, y2, ms=10, label=u'用户2')
    plt.plot(x, yn, ms=10, label=u'新用户')
    plt.legend()  # 让图例生效
    plt.xticks(x, names)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    # plt.xlabel(u"虚拟用户回复次数")  # X轴标签
    # plt.ylabel("聊天热度值")  # Y轴标签
    plt.xlabel(u"每个轮次内迭代次数(iteration)")  # X轴标签
    plt.ylabel("学习率")  # Y轴标签

    plt.show()


if __name__ == '__main__':
    # 准备数据
    s1 = np.random.normal(0, 1, (20, 300))
    s2 = np.random.randn(20, 300)
    u = np.random.randn(5, 300).repeat(2, 0)

    # plot_correlation_2d(s1, s2, u)
    plot_correlation_3d(s1, s2, u)
    # plt_hot()
