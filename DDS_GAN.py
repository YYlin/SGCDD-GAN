# -*- coding: utf-8 -*-
# @Author : YYlin
# @mailbox: ${854280599@qq.com}
# @Time : 2022/3/13 16:07
# @FileName: DDS_GAN.py
from model import Discriminator_No_Classifier
from model import Full_Model
from model import Attention_Unet
from torchvision.utils import save_image
import torch
import os
import torch.nn as nn


class DDS_GAN(object):

    def __init__(self,  Medical_loader_test, config):
        """Initialize configurations."""

        # Data loader.
        self.Medical_loader_test = Medical_loader_test

        # Model configurations.
        self.image_size = config.image_size


        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size

        # Test configurations.
        self.test_iters = config.test_iters
        self.dataset_direction = config.dataset_direction
        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Build the model and tensorboard.
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""
        # Encoder_Generator生成器只是将AGGAN的解码器部分复制了一份attention输出
        self.G = Full_Model()

        self.D = Discriminator_No_Classifier()

        self.S = Attention_Unet()

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.S, 'S')

        if torch.cuda.device_count() > 1:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            self.S = nn.DataParallel(self.S)

        self.G.to(self.device)
        self.D.to(self.device)
        self.S.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def test(self):
        # Load the trained generator.
        self.restore_model(self.test_iters)

        loader_test = self.Medical_loader_test
        with torch.no_grad():
            for i, (x_real, x_real_tumor_mask, x_real_target_b, _, x_real_target_c, _, x_real_name,
                    x_real_label_a, x_real_label_b, x_real_label_c) in enumerate(loader_test):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                x_real_tumor_mask = x_real_tumor_mask.to(self.device)

                if self.dataset_direction == 'nc_pv':
                    x_real_target = x_real_target_b
                else:
                    x_real_target = x_real_target_c

                x_real_target = x_real_target.to(self.device)

                # Translate images.
                fake, attention, content, _ = self.G(x_real)
                # Save the translated images.
                for j in range(len(fake)):
                    result_path = os.path.join(self.result_dir, x_real_name[j])
                    save_image(self.denorm(fake[j].data.cpu()), result_path, nrow=1, padding=0)


                print('Saved real and fake images into {}...'.format(result_path))

