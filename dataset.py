# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate
@File: dataset.py
@Author: QI
@Date: 2021/6/1 11:28
@Description: Dataset、DataLoader相关
"""

import torch
from torch.utils.data import Dataset, DataLoader

from config import args
from utils import Vocab, load_pickle


class MyDataset(Dataset):
    def __init__(self, mode=args.mode, talk=False):
        path_pkl = "./data/" + mode + "_" + str(int(args.cutoff / 10000)) + "w.pkl"
        if talk is True:
            path_pkl = "./dialogue.pkl"
        self.dialogues, self.len_sent, self.name_list, self.users, self.responder, self.target_user, self.bands, self.len_bands, self.response, self.len_response \
            = load_pickle(path_pkl)

    def __getitem__(self, index):
        dig = self.dialogues[index]
        dig_sent_length = self.len_sent[index]
        dig_name_list = self.name_list[index]
        dig_users = self.users[index]
        responder = self.responder[index]
        target_user = self.target_user[index]
        join_sents = self.bands[index]
        join_sents_length = self.len_bands[index]
        response = self.response[index]
        response_length = self.len_response[index]

        return dig, dig_sent_length, dig_name_list, dig_users, responder, target_user, join_sents, join_sents_length, response, response_length

    def __len__(self):
        return len(self.dialogues)


def collate_fn(batch):
    dig, dig_sent_length, dig_name_list, dig_users, responder, target_user, join_sents, join_sents_length, response, response_length = zip(*batch)

    dig = torch.LongTensor(dig).to(args.device)
    dig_sent_length = torch.LongTensor(dig_sent_length).to(args.device)
    dig_users = torch.LongTensor(dig_users).to(args.device)
    responder = torch.LongTensor(responder).to(args.device)
    target_user = torch.LongTensor(target_user).to(args.device)
    join_sents = torch.LongTensor(join_sents).to(args.device)
    join_sents_length = torch.LongTensor(join_sents_length).to(args.device)
    response = torch.LongTensor(response).to(args.device)
    response_length = torch.LongTensor(response_length).to(args.device)
    return dig, dig_sent_length, dig_name_list, dig_users, responder, target_user, join_sents, join_sents_length, response, response_length


def my_dataloader(mode=args.mode, talk=False):
    """
    DataLoader
    :param mode: train/test/dev
    :return: dataloader
    """
    dataset = MyDataset(mode, talk)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True, collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
    # dataloader = my_dataloader(mode="test")
    # for idx, (dig, join_sents, dig_sent_length, dig_name_list, dig_users, responder, target_user, response) in enumerate(dataloader):
    #     print(dig.size())
    #     print(dig_sent_length.size())
    #     print(dig_users.size())
    #     print(dig_name_list)
    #     print(responder)
    #     print(target_user)
    #     print(response.size())
    #     break
    pass
