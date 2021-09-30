# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate
@File: dataset_process.py
@Author: QI
@Date: 2021/5/31 14:51
@Description: 处理原数据集(dev/test/train.txt),清除数据集中不规范的样本
"""


def clean_data(file):
    """
    清除数据集中不规范的样本
    :param file: 原数据集(dev/test/train.txt)
    :return: 清理后的样本数据集list
    """
    with open(file, "r", encoding='utf-8') as f:
        data = f.read()
        dialogs = data.split("\r\n\r\n") if "\r\n" in data else data.split("\n\n")  # 样本集List

        for dialog in dialogs:
            lines = dialog.split("\n")
            if len(lines) < 6:  # 样本不足6行，移除
                dialogs.remove(dialog)
                continue
            for i in range(len(lines)):  # 每行不是四元组或者每行Speaker==Addressee，移除
                if len(lines[i].split("\t")) != 4 or lines[i].split("\t")[1] == lines[i].split("\t")[2]:
                    dialogs.remove(dialog)
                    break

    print("clean SUCCESS!")
    return dialogs


def rewrite_data(cleaned_dialogs_list, file):
    """
    清理后的样本数据集list写入文件
    :param cleaned_dialogs_list: 清理后的样本数据集list
    :param file: 写入的文件
    :return:
    """
    with open(file, "w", encoding="utf-8") as f:
        for index, dialog in enumerate(cleaned_dialogs_list):
            if index != 0:
                f.write("\n\n")
            f.write(dialog)
    print("rewrite SUCCESS!")


if __name__ == '__main__':
    datasets = ["dev.txt", "test.txt", "train.txt"]
    for dataset in datasets:
        cleaned_data = clean_data(dataset)
        rewrite_data(cleaned_data, "../processed/cleaned_" + dataset)
