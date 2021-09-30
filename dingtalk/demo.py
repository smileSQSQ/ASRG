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
Vuser = DingtalkChatbot(webhook, secret=secret)


def sleep_time(hour, minute, sec):
    return hour * 3600 + minute * 60 + sec


def id_name(members):
    '''
    建立openId和groupNick之间的映射
    :param members: [{'FullName': '', 'groupNick': '', 'role': '', 'openId': },{'FullName': '', 'groupNick': '', 'role': '', 'openId': }......]
    :return: dict
    '''
    id2name = {}
    name2id = {}
    for item in members:
        name = item['groupNick']
        openid = str(item['openId'])
        id2name[openid] = name
        name2id[name] = openid
    return id2name, name2id


def write_messages(messages, id2name):
    '''
    获取聊天信息,按格式写入文件
    :param messages: [['messageData', timestamp, openID, 'nick'],['messageData', timestamp, openID, 'nick'],['messageData', timestamp, openID, 'nick']]
    :return:
    '''
    last_messages = messages[-5:]
    f = open("./dialogue.txt", "w", encoding="utf-8")
    for item in last_messages:
        timestamp = str(item[1])
        sender = item[3]
        message = item[0]
        if message.find('@') == -1:
            utterance = message.replace('\n', '').replace('\r', '')
            addressee = '-'
        else:
            utterance, addressee = message.split('@')
            utterance = utterance.replace('\n', '').replace('\r', '')
            addressee = id2name.get(addressee, '-')

        f.write(timestamp + "\t" + sender + "\t" + addressee + "\t" + utterance + "\n")

    # 构造第六行
    timestamp = str(time.time()).split('.')[0]
    # sender = openid2name.get(dt.getOpenId())
    sender = "虚拟用户"
    addressee = "target"
    utterance = "Waiting to be generated"
    f.write(timestamp + "\t" + sender + "\t" + addressee + "\t" + utterance + "\n")

    f.close()


def sample1():
    Vuser.send_text(msg="Update the kernel version and try to solve it", at_mobiles=['15121079110'])


def sample2():
    Vuser.send_text(msg="Yes it may be a driver error", at_mobiles=['19821810536'])


def sample3():
    Vuser.send_text(msg="Try other version of the graphics card driver", at_mobiles=['15121079110'])


def talk(cid, threshold):
    # 获取群员信息,建立openId和FullName之间的映射,FullName格式:昵称_手机号
    members = dt.getUsers(cid)
    members = dt.formatMembersInfo(members)
    id2name, name2id = id_name(members)

    # 获取聊天信息,按格式写入
    messages = dt.getMessage(cid)
    write_messages(messages, id2name)

    # 调用模型生成
    model = ASRGMPC().to(args.device)
    target_addressee, target_message, logit = interact(model)
    print("虚拟用户->" + target_addressee + "      " + target_message)

    if logit > threshold:
        # 发送信息
        # dt.send(cid, target_message + "@" + target_addressee)
        if target_addressee == "-":
            Vuser.send_text(msg=target_message, is_at_all=True)
        else:
            at_mobiles = [target_addressee.split('_')[1]]
            Vuser.send_text(msg=target_message, at_mobiles=at_mobiles)
    else:
        print("无合适回复对象，本次不发言,最高概率=" + logit)


if __name__ == '__main__':
    cid = cid_dict['新手体验群']  # 群名称
    threshold = 0.7

    time.sleep(180)
    sample1()
    time.sleep(120)

    sample2()
    time.sleep(40)

    sample3()

    # while True:
    #     sleep_time(0, 0, 45)
    #     talk(cid, threshold)
