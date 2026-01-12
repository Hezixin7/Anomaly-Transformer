import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True # 自动寻找最适合当前硬件配置的卷积算法
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config)) # 初始化

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4) # 学习率
    parser.add_argument('--num_epochs', type=int, default=10) # 训练轮次
    parser.add_argument('--k', type=int, default=3) # 异常分数
    parser.add_argument('--win_size', type=int, default=100) # 窗口大小
    parser.add_argument('--input_c', type=int, default=38) # 输入特征通道数
    parser.add_argument('--output_c', type=int, default=38) # 输出特征通道数
    parser.add_argument('--batch_size', type=int, default=1024) # 批次大小
    parser.add_argument('--pretrained_model', type=str, default=None) # 与训练模型路径
    parser.add_argument('--dataset', type=str, default='credit') # 数据集名字
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test']) # 运行模式
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv') # 数据文件路径
    parser.add_argument('--model_save_path', type=str, default='checkpoints') # 模型保存路径
    parser.add_argument('--anormly_ratio', type=float, default=4.00) # 异常比例

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
