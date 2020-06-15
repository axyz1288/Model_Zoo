#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        
        def block(input_dim, output_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(input_dim, output_dim, 5, stride=2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.CELU()
            )
        
        self.layer1 = nn.Sequential(nn.Linear(g_input_size, 1024 * 4 * 4),
                                    nn.BatchNorm1d(1024 * 4 * 4),
                                    nn.Tanh())
        self.layer2 = block(1024, 512)
        self.layer3 = block(512, 256)
        self.layer4 = block(256, 128) 
        self.layer5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 3, 5, stride=1, padding=2),
            nn.BatchNorm2d(3),
            nn.Sigmoid()) # 64x64
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x.view(-1, 1024, 4, 4))
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def block(input_dim, output_dim):
            return nn.Sequential(
                nn.Conv2d(input_dim, output_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(0.2)
            )
        
        self.layer1 = block(3, 128) 
        self.layer2 = block(128, 256)
        self.layer3 = block(256, 512)
        self.layer4 = block(512, 1024)
        self.layer5 = block(1024, g_input_size)
        self.layer6 = nn.Sequential(nn.Linear(g_input_size * 2 * 2, 1),
                                    nn.Sigmoid())
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x.view(-1, g_input_size * 2 * 2))
        return x