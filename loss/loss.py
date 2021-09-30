import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker


def loss_data_old(file):
    """
    得到loss数据
    """
    f_out = open("./loss_origin_old.txt", "a+", encoding="utf-8")

    with open(file, "r", encoding='utf-8') as f:
        data = f.read()
        lines = data.split("\n")

        for line in lines:
            if "(" in line:
                f_out.write(line.split("(")[1])
                f_out.write("\n")


def loss_data_txt_old(file):
    """
    得到最终loss数据
    """
    f_out = open("./loss_old.txt", "a+", encoding="utf-8")

    with open(file, "r", encoding='utf-8') as f:
        data = f.read()
        lines = data.split("\n")

        for line in lines:
            temp = line.split(")")[0]
            temp = temp.split(", ")
            loss1 = temp[0].split("=")[1]
            loss2 = temp[1].split("=")[1]
            f_out.write(loss1 + "," + loss2)
            f_out.write("\n")


def loss_data(file):
    """
    得到loss数据
    """
    f_out = open("./loss_origin.txt", "a+", encoding="utf-8")

    with open(file, "r", encoding='utf-8') as f:
        data = f.read()
        lines = data.split("\n")

        for line in lines:
            if "Instant:" in line:
                f_out.write(line.split("Instant:")[1])
                f_out.write("\n")


def loss_data_txt(file):
    """
    得到最终loss数据
    """
    f_out = open("./loss.txt", "a+", encoding="utf-8")

    with open(file, "r", encoding='utf-8') as f:
        data = f.read()
        lines = data.split("\n")

        for line in lines:
            temp = line.split("(")[1]
            temp = temp.split(")")[0]
            temp = temp.split(", ")
            loss1 = temp[0].split("=")[1]
            loss2 = temp[1].split("=")[1]
            f_out.write(loss1 + "," + loss2)
            f_out.write("\n")


def loss_image(file):
    sum_loss = []
    select_loss = []
    generate_loss = []
    with open(file, "r", encoding='utf-8') as f:
        data = f.read()
        lines = data.split("\n")

        for line in lines:
            loss1 = line.split(",")[0]
            loss2 = line.split(",")[1]
            select_loss.append(np.float(loss1))
            generate_loss.append(np.float(loss2))
            sum_loss.append(np.float(loss1) + np.float(loss2))

    # 绘图
    fig = plt.figure()
    xs = np.arange(len(sum_loss))
    plt.yticks(np.arange(0, 12, 1))
    plt.xlabel("Epoch")  # X轴标签
    plt.plot(xs, sum_loss, color='r', label="总损失")
    plt.plot(xs, select_loss, color='coral', label="选择模型损失")
    plt.plot(xs, generate_loss, color='g', label="生成模型损失")
    plt.legend()
    plt.show()
    # plt.savefig("loss.png")


if __name__ == '__main__':
    # loss_data("./nohup.out")
    # loss_data_txt("./loss_origin.txt")
    # loss_image("./loss.txt")

    # loss_data_old("./nohup_train.out")
    # loss_data_txt_old("./loss_origin_old.txt")

    fig = plt.figure()
    xs = [0, 20, 40, 60, 80, 100]
    plt.yticks(np.arange(0, 12, 1))
    plt.xlabel("Epoch")  # X轴标签
    plt.plot(xs, [1, 2, 3, 4, 5, 6], color='r', label="total loss")
    plt.legend()
    plt.show()
