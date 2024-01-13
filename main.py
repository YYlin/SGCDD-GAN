# -*- coding: utf-8 -*-
# @Author : YYlin
# @mailbox: ${854280599@qq.com}
# @Time : 2022/2/10 20:16
# @FileName: main.py
import os
import argparse
from data_loader import load_test_data
from torch.backends import cudnn
from DDS_GAN import DDS_GAN


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    image_dir = os.path.join('../dataset', config.dataset)
    config.model_save_dir = 'Pretrained-Model/'
    config.result_dir = 'Result'

    # Create directories if not exist.

    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    Medical_loader_test = load_test_data(image_dir, config.image_size, config.batch_size,
                                         config.mode, config.num_workers)

    # Solver for training and testing StarGAN.
    solver = DDS_GAN(Medical_loader_test, config)
    solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=256, help='image resolution')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='MICCAI_Dataset')
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

    # Directories.F
    parser.add_argument('--model_save_dir', type=str, default='ddgan/models')
    parser.add_argument('--result_dir', type=str, default='ddgan/results')

    # Step size.
    parser.add_argument('--dataset_direction', type=str, default='nc_art', choices=['nc_pv', 'nc_art'])

    config = parser.parse_args()

    print(config)
    main(config)




