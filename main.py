# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate
@File: main.py
@Author: QI
@Date: 2021/6/7 16:46
@Description: 模型训练、测试主程序
"""

import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from nlgeval import compute_metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import args
from dataset import my_dataloader
from metrics.metrics import cal_acc_batch, cal_acc, print_loss
from utils import Vocab, load_pickle, build_model_output, build_hyp
from model.combine_model import ASRGMPC

warnings.filterwarnings('ignore')

model = ASRGMPC().to(args.device)
writer = SummaryWriter()
# model = nn.DataParallel(model, device_ids=[0, 1], output_device=0)

if args.select_parameters_fix:
    for para in model.adr_select.parameters():
        para.requires_grad = False

# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-6, last_epoch=-1)
loss_fn1 = nn.CrossEntropyLoss()
loss_fn2 = nn.CrossEntropyLoss(ignore_index=Vocab.PAD)


def train():
    model.train()
    if args.use_pretrain_tf:
        model.transformer.load_state_dict(torch.load("./preTrain/save/PreTrain_model.pt", map_location=args.device))
    if args.use_pretrain_selection:
        model.adr_select.load_state_dict(torch.load("./save/select_model" + str(args.select_way) + ".pt", map_location=args.device))

    for epo in range(args.epoch):
        count = 0
        running_loss1, running_loss2, running_loss = 0.0, 0.0, 0.0
        epoch_loss1, epoch_loss2, epoch_loss = 0.0, 0.0, 0.0

        train_dataloader = my_dataloader(mode=args.mode)
        for idx, (dig, dig_sent_length, dig_name_list, dig_users, responder, target_user, join_sents, join_sents_length, response, response_length) in \
                tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='train:'):

            optimizer.zero_grad()

            select_output, generate_output, _, _ = model(dig, dig_sent_length, dig_users, responder, join_sents, join_sents_length, response[:, :-1], response_length, target_user)
            # select_output:[batch_size,max_roles-1], generate_output:[batch_size * tgt_max_len, vocab_size]
            index, right = cal_acc_batch(select_output, target_user)
            count += right

            loss1 = loss_fn1(select_output, target_user)
            loss2 = loss_fn2(generate_output, response[:, 1:].contiguous().view(-1))
            loss = args.lamda1 * loss1 + args.lamda2 * loss2

            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            running_loss += loss.item()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            epoch_loss += loss.item()

            if (idx + 1) % args.print_steps == 0:
                print_loss(idx, epo, loss, loss1, loss2, running_loss, running_loss1, running_loss2)
                running_loss1, running_loss2, running_loss = 0.0, 0.0, 0.0

            loss.backward()
            optimizer.step()

        acc = cal_acc(count, (len(train_dataloader) * args.batch_size))
        print(f'\nEpoch{epo + 1}, acc={acc}')

        writer.add_scalars("Epoch_loss", {"epoch_loss1": epoch_loss1, "epoch_loss2": epoch_loss2, "epoch_loss": epoch_loss}, epo)
        torch.save(model.state_dict(), "./save/model_" + str(args.select_way) + str(args.generate_way) + ".pt")

        scheduler.step()

    torch.save(model.state_dict(), "./save/model_" + str(args.select_way) + str(args.generate_way) + ".pt")
    writer.close()


@torch.no_grad()
def val():
    model.eval()
    args.mode = "dev"
    args.shuffle = False
    args.batch_size = 1

    vocab = load_pickle(r"./data/vocab_" + str(int(args.cutoff / 10000)) + "w.pkl")
    f_out = open("./save/out_dev.txt", "a+", encoding="utf-8")
    f_hyp = open("./metrics/hyp_dev.txt", "a+", encoding="utf-8")
    model.load_state_dict(torch.load("./save/model_" + str(args.select_way) + str(args.generate_way) + ".pt", map_location=args.device))

    count = 0
    dev_dataloader = my_dataloader(mode=args.mode)
    for idx, (dig, dig_sent_length, dig_name_list, dig_users, responder, target_user, join_sents, join_sents_length, response, response_length) in \
            tqdm(enumerate(dev_dataloader), total=len(dev_dataloader), desc='dev'):

        select_output, generate_output, a_res, a_tgt = model(dig, dig_sent_length, dig_users, responder, join_sents, join_sents_length,
                                                             response[:, :-1], response_length, target_user)
        # select_output:[batch_size,max_roles-1], generate_output:[batch_size * tgt_max_len, vocab_size]
        index, right = cal_acc_batch(select_output, target_user)
        count += right
        cur_acc = cal_acc(count, ((idx + 1) * args.batch_size))
        if (idx + 1) % 1000 == 0:
            print("\ncur_acc=", cur_acc)

        # 生成回复
        generate_response = model.transformer.generate(join_sents, join_sents_length, a_res, a_tgt)
        generate_response = vocab.digits2words(generate_response)

        build_model_output(dig_name_list[0][responder.item()], dig_name_list[0][index.item()], generate_response, f_out)
        build_hyp(generate_response, f_hyp)

    acc = cal_acc(count, (len(dev_dataloader) * args.batch_size))
    print("\nacc=", acc)

    f_out.close()
    f_hyp.close()

    metrics_dict = compute_metrics(hypothesis='metrics/hyp_dev.txt', references=['metrics/ref_dev.txt'])
    print(metrics_dict)


@torch.no_grad()
def test():
    model.eval()
    args.mode = "test"
    args.shuffle = False
    args.batch_size = 1

    vocab = load_pickle(r"./data/vocab_" + str(int(args.cutoff / 10000)) + "w.pkl")
    f_out = open("./save/out_test.txt", "a+", encoding="utf-8")
    f_hyp = open("./metrics/hyp_test.txt", "a+", encoding="utf-8")
    model.load_state_dict(torch.load("./save/model_" + str(args.select_way) + str(args.generate_way) + ".pt", map_location=args.device))

    count = 0
    test_dataloader = my_dataloader(mode=args.mode)
    for idx, (dig, dig_sent_length, dig_name_list, dig_users, responder, target_user, join_sents, join_sents_length, response, response_length) in \
            tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='test:'):

        select_output, generate_output, a_res, a_tgt = model(dig, dig_sent_length, dig_users, responder, join_sents, join_sents_length,
                                                             response[:, :-1], response_length, target_user)
        # select_output:[batch_size,max_roles-1], generate_output:[batch_size * tgt_max_len, vocab_size]
        index, right = cal_acc_batch(select_output, target_user)
        count += right
        cur_acc = cal_acc(count, ((idx + 1) * args.batch_size))
        if (idx + 1) % 1000 == 0:
            print("\ncur_acc=", cur_acc)

        # 生成回复
        generate_response = model.transformer.generate(join_sents, join_sents_length, a_res, a_tgt)
        generate_response = vocab.digits2words(generate_response)

        build_model_output(dig_name_list[0][responder.item()], dig_name_list[0][index.item()], generate_response, f_out)
        build_hyp(generate_response, f_hyp)

    acc = cal_acc(count, (len(test_dataloader) * args.batch_size))
    print("\nacc=", acc)

    f_out.close()
    f_hyp.close()

    metrics_dict = compute_metrics(hypothesis='metrics/hyp_test.txt', references=['metrics/ref_test.txt'])
    print(metrics_dict)


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "dev":
        val()
    elif args.mode == "test":
        test()
    else:
        print("args.mode is ERROR!")
