# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate
@File: utils.py
@Author: QI
@Date: 2021/5/31 15:27
@Description: 词表构建、序列化数据集及其他功能函数
"""

import nltk
import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter

from config import args


class Vocab:
    PAD_TAG, UNK_TAG, SOS_TAG, EOS_TAG, SEP_TAG = "PAD", "UNK", "SOS", "EOS", "SEP"  # 特殊符号定义
    PAD, UNK, SOS, EOS, SEP = 0, 1, 2, 3, 4

    def __init__(self):
        """
        词表三大件
        """
        self.w2idx = {self.PAD_TAG: self.PAD,
                      self.UNK_TAG: self.UNK,
                      self.SOS_TAG: self.SOS,
                      self.EOS_TAG: self.EOS,
                      self.SEP_TAG: self.SEP}  # dict
        self.idx2w = dict(zip(self.w2idx.values(), self.w2idx.keys()))  # # dict
        self.pretrained_embeddings = None  # [len(w2idx),dim]

    def build_vocab(self, files, pretrained_vec_file, cutoff=args.cutoff, use_pretrained_embeddings=args.use_pretrained_embeddings):
        """
        构建词表(w2idx/idx2w/pretrained_embeddings)
        :param files: [cleaned_train.txt, cleaned_dev.txt, cleaned_test.txt]
        :param pretrained_vec_file: "wiki_300d.vec"
        :param cutoff: 词表中词的个数
        :param use_pretrained_embeddings: 是否加载预训练的词向量
        :return: 词表序列化文件
        """
        all_words = []
        for item, file in enumerate(files):
            if file != "./data/preTrain.txt":  # cleaned_train.txt, cleaned_dev.txt, cleaned_test.txt
                with open(file, "r", encoding='utf-8') as f:
                    for line in tqdm(f.readlines(), desc="building_vocab_" + str(item + 1) + "_" + file):
                        if line in ['\n', '\r\n']:
                            continue
                        utterance = line.split("\t")[3].lower().strip()
                        words = nltk.word_tokenize(utterance)
                        all_words.extend(words)
            else:  # preTrain.txt
                with open(file, "r", encoding='utf-8') as f:
                    for line in tqdm(f.readlines(), desc="building_vocab_" + str(item + 1) + "_" + file):
                        utterance, response = line.split("\t")
                        words = nltk.word_tokenize(utterance.lower().strip())
                        all_words.extend(words)
                        words = nltk.word_tokenize(response.lower().strip())
                        all_words.extend(words)

        statistics_words = Counter(all_words)  # Counter统计词频，(word, count)
        print(f'所有数据集中共含词: {len(statistics_words)}')
        common_words = statistics_words.most_common(cutoff)

        for index, (word, count) in enumerate(common_words):
            self.w2idx[word] = index + 5
            self.idx2w[index + 5] = word

        if use_pretrained_embeddings:
            with open(pretrained_vec_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
                count, dim = map(int, f.readline().split())  # 词向量文件第一行记录词数以及维度
                self.pretrained_embeddings = np.random.random([len(self.w2idx), dim])
                self.pretrained_embeddings[0, :] = np.zeros([1, dim])  # "PAD"全0
                for line in f:
                    tokens = line.rstrip().split()
                    w, e = tokens[0], tokens[1:]
                    if w in self.w2idx.keys():
                        index = self.w2idx[w]
                        self.pretrained_embeddings[index, :] = e
                self.pretrained_embeddings[1, :] = np.mean(self.pretrained_embeddings, 0)  # "UNK"为全部词的平均

        with open("./data/vocab_" + str(int(args.cutoff / 10000)) + "w.pkl", 'wb') as f:
            pickle.dump(self, f)
        print(f'词表已保存，词表大小：{len(self.w2idx)}')

    def words2digits(self, sentence, max_len, mode="encoder"):
        """
        句子转化为数字
        :param sentence: 句子单词list
        :param max_len: 单个句子的最大长度
        :param mode: encoder/decoder不同模式不同的处理方式
        :return: 数字list
        """
        if mode == "encoder":  # encoder中不需要添加SOS和EOS
            sentence = [self.w2idx.get(w, self.UNK) for w in nltk.word_tokenize(sentence)]
            sentence = sentence[-max_len:] if len(sentence) > max_len else sentence
            sentence = sentence + [self.PAD] * (max_len - len(sentence)) if len(sentence) < max_len else sentence
        elif mode == "decoder":
            sentence = [self.SOS] + [self.w2idx.get(w, self.UNK) for w in nltk.word_tokenize(sentence)] + [self.EOS]
            sentence = [self.SOS] + sentence[-(max_len - 1):] if len(sentence) > max_len else sentence
            sentence = sentence + [self.PAD] * (max_len - len(sentence)) if len(sentence) < max_len else sentence
        return sentence

    def digits2words(self, indices):
        """
        数字转化为句子
        :param indices: 数字list
        :return: 句子单词list
        """
        result = []
        for i in indices:
            if self.idx2w.get(int(i)) == self.EOS_TAG:
                break
            result.append(self.idx2w.get(int(i)))
        return result

    def __len__(self):
        """
        词表长度
        :return: len(w2idx)
        """
        return len(self.w2idx)


def load_pickle(file):
    """
    加载序列化文件
    :param file: 序列化的文件
    :return:
    """
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj


def sentence_length(sent, max_len):
    """
    获取句子长度
    :param sent: list
    :param max_len: 句子最大长度
    :return: len(sent)
    """
    return len(sent) if len(sent) < max_len else max_len


def load_data(dataset):
    """
    序列化数据集
    :param dataset: [cleaned_train.txt, cleaned_dev.txt, cleaned_test.txt]
    :return: 数据集序列化文件
    """
    vocab = load_pickle(r"./data/vocab_" + str(int(args.cutoff / 10000)) + "w.pkl")  # 加载词表

    with open(dataset, encoding='utf-8') as f:
        raw_data = f.read()
        data_list = raw_data.split("\r\n\r\n") if "\r\n" in raw_data else raw_data.split("\n\n")  # 以空行分割

        dialogues, len_sent = [], []
        name_list, users, responder, target_user = [], [], [], []
        bands, len_bands = [], []
        response, len_response = [], []

        for dig in tqdm(data_list, desc=args.mode):  # 处理每段对话
            c_dialogues, c_ls = [], []
            c_nl, c_users = [], []
            c_bands, c_lb, = "", []

            lines = dig.split("\n")
            for index, line in enumerate(lines):  # 处理每段对话的每一行
                if index < args.context_window:  # input
                    spk = line.split('\t')[1]
                    adr = line.split('\t')[2]
                    sent = (line.split('\t')[3]).lower().strip()

                    c_ls.append(sentence_length(sent, args.input_max_len))  # [context_window]

                    sent_ids = vocab.words2digits(sent, args.input_max_len, "encoder")
                    c_dialogues.append(sent_ids)  # [context_window,input_max_len]

                    sent = sent + " " + vocab.SEP_TAG + " " if index != 4 else sent
                    c_bands += sent

                    if spk not in c_nl:
                        c_nl.append(spk)  # [max_roles]
                    if adr not in c_nl:
                        c_nl.append(adr)

                    spk_id = c_nl.index(spk)
                    adr_id = c_nl.index(adr)
                    c_users.append([spk_id, adr_id])  # [context_window,2]
                else:  # target
                    speaker = line.split('\t')[1]
                    addressee = line.split('\t')[2]
                    res = (line.split('\t')[3]).lower().strip()

                    lr = sentence_length(res, args.target_max_len)
                    lr = lr + 2 if lr <= args.target_max_len - 2 else args.target_max_len  # SOS、EOS计入长度中
                    len_response.append(lr)  # [dig_count]

                    response_ids = vocab.words2digits(res, args.target_max_len, "decoder")
                    response.append(response_ids)  # [dig_count,target_max_len]

                    if speaker not in c_nl:
                        c_nl.append(speaker)
                    if addressee not in c_nl:
                        c_nl.append(addressee)
                    while len(c_nl) < args.max_roles:
                        c_nl.append(" ")

                    speaker_id = c_nl.index(speaker)
                    addressee_id = c_nl.index(addressee)
                    responder.append(speaker_id)  # [dig_count]
                    target_user.append(addressee_id)  # [dig_count]

            len_bands.append(sentence_length(c_bands, args.sents_max_len))  # [dig_count]
            c_bands = vocab.words2digits(c_bands, args.sents_max_len, "encoder")
            bands.append(c_bands)
            name_list.append(c_nl)
            users.append(c_users)
            len_sent.append(c_ls)
            dialogues.append(c_dialogues)

    with open("./data/" + args.mode + "_" + str(int(args.cutoff / 10000)) + "w.pkl", 'wb') as f:
        pickle.dump((dialogues, len_sent, name_list, users, responder, target_user, bands, len_bands, response, len_response), f)
    # dialogues: [dig_count, window, input_max_len]
    # len_sent: [dig_count, window]
    # name_list: [dig_count, max_roles]
    # users: [dig_count, window, 2]
    # responder: [dig_count]
    # target_user: [dig_count]
    # bands: [dig_count, sents_max_len]
    # len_bands: [dig_count]
    # response: [dig_count, target_max_len]
    # len_response: [dig_count]
    return dialogues, len_sent, name_list, users, responder, target_user, bands, len_bands, response, len_response


def load_data_talk(dataset, vocab):
    """
    序列化数据集
    :param dataset: dialogue.txt
    :return: 数据集序列化文件
    """

    with open(dataset, encoding='utf-8') as f:
        raw_data = f.read()
        data_list = raw_data.split("\r\n\r\n") if "\r\n" in raw_data else raw_data.split("\n\n")  # 以空行分割

        dialogues, len_sent = [], []
        name_list, users, responder, target_user = [], [], [], []
        bands, len_bands = [], []
        response, len_response = [], []

        for dig in data_list:  # 处理每段对话
            c_dialogues, c_ls = [], []
            c_nl, c_users = [], []
            c_bands, c_lb, = "", []

            lines = dig.split("\n")
            for index, line in enumerate(lines):  # 处理每段对话的每一行
                if index < args.context_window:  # input
                    spk = line.split('\t')[1]
                    adr = line.split('\t')[2]
                    sent = (line.split('\t')[3]).lower().strip()

                    c_ls.append(sentence_length(sent, args.input_max_len))  # [context_window]

                    sent_ids = vocab.words2digits(sent, args.input_max_len, "encoder")
                    c_dialogues.append(sent_ids)  # [context_window,input_max_len]

                    sent = sent + " " + vocab.SEP_TAG + " " if index != 4 else sent
                    c_bands += sent

                    if spk not in c_nl:
                        c_nl.append(spk)  # [max_roles]
                    if adr not in c_nl:
                        c_nl.append(adr)

                    spk_id = c_nl.index(spk)
                    adr_id = c_nl.index(adr)
                    c_users.append([spk_id, adr_id])  # [context_window,2]
                elif index == args.context_window:  # target
                    line = line.split('\t')
                    speaker = line[1]
                    addressee = line[2]
                    res = (line[3]).lower().strip()

                    lr = sentence_length(res, args.target_max_len)
                    lr = lr + 2 if lr <= args.target_max_len - 2 else args.target_max_len  # SOS、EOS计入长度中
                    len_response.append(lr)  # [dig_count]

                    response_ids = vocab.words2digits(res, args.target_max_len, "decoder")
                    response.append(response_ids)  # [dig_count,target_max_len]

                    if speaker not in c_nl:
                        c_nl.append(speaker)
                    if addressee not in c_nl:
                        c_nl.append(addressee)
                    while len(c_nl) < args.max_roles:
                        c_nl.append(" ")

                    speaker_id = c_nl.index(speaker)
                    addressee_id = c_nl.index(addressee)
                    responder.append(speaker_id)  # [dig_count]
                    target_user.append(addressee_id)  # [dig_count]

            len_bands.append(sentence_length(c_bands, args.sents_max_len))  # [dig_count]
            c_bands = vocab.words2digits(c_bands, args.sents_max_len, "encoder")
            bands.append(c_bands)
            name_list.append(c_nl)
            users.append(c_users)
            len_sent.append(c_ls)
            dialogues.append(c_dialogues)

    with open(r"./dialogue.pkl", 'wb') as f:
        pickle.dump((dialogues, len_sent, name_list, users, responder, target_user, bands, len_bands, response, len_response), f)

    return dialogues, len_sent, name_list, users, responder, target_user, bands, len_bands, response, len_response


def build_refs(mode="dev"):
    """
    建立评价参考句子数据文件(每个样本最后一行)
    :param mode: dev/test
    :return: 参考句子数据文件
    """
    dataset = "./data/unprocessed/dev.txt" if mode == "dev" else "./data/unprocessed/test.txt"

    fout = open("./metrics/ref_" + mode + ".txt", "a+", encoding="utf-8")

    with open(dataset, encoding='utf-8') as f:
        raw_data = f.read()
        data_list = raw_data.split("\r\n\r\n") if "\r\n" in raw_data else raw_data.split("\n\n")

        for dig in data_list:  # 处理每段对话
            lines = dig.split("\n")
            for index, line in enumerate(lines):  # 处理每段对话的每一行
                sent = (line.split('\t')[3]).lower().strip()
                if (index + 1) == 6:
                    fout.write(sent + "\n")
    fout.close()


def build_hyp(generate_response, f):
    """
    建立待评价句子数据文件
    :param generate_response: 生成的句子单词list
    :param f: 输出流
    :return: 待评价句子数据文件
    """
    f.write(" ".join(generate_response) + "\n")


def build_model_output(res, adr, generate_response, f):
    """
    建立模型输出结果文件(三元组)
    :param res: speaker
    :param adr: addressee
    :param generate_response: 生成的句子单词list
    :param f: 输出流
    :return: 模型输出结果文件
    """
    f.write(res + "\t" + adr + "\t")
    f.write(" ".join(generate_response) + "\n")


if __name__ == "__main__":
    vocabulary = Vocab()
    vocabulary.build_vocab(["./data/unprocessed/train.txt", "./data/unprocessed/dev.txt", "./data/unprocessed/test.txt"],
                           "./data/wiki_300d.vec", args.cutoff, args.use_pretrained_embeddings)

    for m in ["train", "dev", "test"]:
        args.mode = m
        load_data("./data/unprocessed/" + args.mode + ".txt")

    for m in ["dev", "test"]:
        build_refs(m)
