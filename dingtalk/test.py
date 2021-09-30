# -*- coding: UTF-8 -*-
"""
@Project: SelectGenerate 
@File: test.py
@Author: QI
@Date: 2021/9/8 16:45 
@Description: None
"""

from dingtalkchatbot.chatbot import DingtalkChatbot

from dingtalk.talk_model import interact
from dingtalk.talk_utils import Work
from utils import Vocab
from model.combine_model import ASRGMPC
from config import args
import time

dt = Work()
dt.Login()
cid_dict = dt.getConversation()

# WebHook地址
webhook = 'https://oapi.dingtalk.com/robot/send?access_token=f5a40e7ce2da3bba30cee6dcb90ccd56306deeed56925dd743703f228621f646'
secret = 'SEC706207ace6d762ef1eaa087a49b2b6b2ac1583fcd9e77d603a86c01b53595486'  # 可选：创建机器人勾选“加签”选项时使用
# 初始化机器人小丁
Vuser = DingtalkChatbot(webhook, secret=secret)  # 方式二：勾选“加签”选项时使用（v1.5以上新功能）


def main():
    Vuser.send_text(msg="Yes, you can find it on ubuntu forum", at_mobiles=[15055101615])


if __name__ == '__main__':
    main()
