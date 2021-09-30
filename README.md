# SelectGenerate

Addressee Selection and Response Generation on Multi-Party Chatbot

## File Description

```
config.py: 参数设置
dataset.py: Dataset、Dataloader相关
main.py: 主程序-训练、测试入口
utils.py: 词表构建、数据集加载、其他杂项函数
select model:
    0  -- max pooling
    1  -- conv
    2  -- gru
    3  -- luong attn
    4  -- self attn
generate model:
    generate_api_0.py  -- 使用Pytorch自带Transformer相关API实现的原版Transformer
    generate_api_1.py  -- Transformer + A_res(In Decoder after multi_head_attn) + A_tgt(In Decoder after multi_head_attn)
    generate_api_2.py  -- Transformer + A_res(In Encoder) + A_tgt(In Decoder before multi_head_attn)
    generate_api_3.py  -- Transformer + A_tgt(In Decoder parallel with multi_head_attn)
```

## Dataset

Ubuntu IRC chat log - [项目名称](网址)

## Prerequisites

- Python3.6
- PyTorch = 1.4.0
- nlg-eval = 2.3
- nltk = 3.5
- tensorboard = 2.4.0
- numpy
- matplotlib
- sklearn
- argparse
- pickle
- collections

## Implementation Notes

- Full vocabs were used instead of using only 50000 common words as the paper did.
- Adaptive softmax were adopted instead of cross entropy in order to speed up training process.
- The model was trained on one 1080 ti, and it took 2 days for 100 epochs.
- Beam Search method is not parallel.

## Usage

1. Run ```data/unprocessed/dataset_process.py``` to process raw data files.
2. Run ```utils.py``` to create Vocab and load dataset files.
3. Run ```main.py``` to train the model.

## Generated Examples

    待补充

## Todo

    交互接口


