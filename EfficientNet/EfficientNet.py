#!/usr/bin/env python
# coding: utf-8

# In[1]:

from torch import nn
import torch


# In[2]:


arg_blocks = [
    # stage 2
    {'input_dim': 32, 'output_dim': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'expansion_ratio': 1,
     'se_ratio': 0.25, 'bias': False, 'dropout': 0.2, 'repeat_times': 1},
    # stage 3
    {'input_dim': 16, 'output_dim': 24, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'expansion_ratio': 6,
     'se_ratio': 0.25, 'bias': False, 'dropout': 0.2, 'repeat_times': 2},
    # stage 4
    {'input_dim': 24, 'output_dim': 40, 'kernel_size': 5, 'stride': 2, 'padding': 2, 'expansion_ratio': 6,
     'se_ratio': 0.25, 'bias': False, 'dropout': 0.2, 'repeat_times': 2},
    # stage 5
    {'input_dim': 40, 'output_dim': 80, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'expansion_ratio': 6,
     'se_ratio': 0.25, 'bias': False, 'dropout': 0.2, 'repeat_times': 3},
    # stage 6
    {'input_dim': 80, 'output_dim': 112, 'kernel_size': 5, 'stride': 1, 'padding': 2, 'expansion_ratio': 6,
     'se_ratio': 0.25, 'bias': False, 'dropout': 0.2, 'repeat_times': 3},
    # stage 7
    {'input_dim': 112, 'output_dim': 192, 'kernel_size': 5, 'stride': 2, 'padding': 2, 'expansion_ratio': 6,
     'se_ratio': 0.25, 'bias': False, 'dropout': 0.2, 'repeat_times': 4},
    # stage 8
    {'input_dim': 192, 'output_dim': 320, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'expansion_ratio': 6,
     'se_ratio': 0.25, 'bias': False, 'dropout': 0.2, 'repeat_times': 1},
]


# #  MBConv

# In[3]:


class MBConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, 
                 stride, 
                 padding,
                 expansion_ratio=6,
                 se_ratio=0.25,
                 bias=False,
                 dropout=0.2):
        super(MBConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.expansion_ratio = expansion_ratio
        self.stride = stride

        # Expansion Convolution
        bottleNeck_dim = int(expansion_ratio * input_dim)
        self.exp1 = nn.Conv2d(self.input_dim, bottleNeck_dim, 1, stride=1, bias=bias)
        self.exp2 = nn.BatchNorm2d(bottleNeck_dim)
        self.exp3 = nn.ReLU6()

        # Depthwise Convolution
        self.dconv1 = nn.Conv2d(bottleNeck_dim, bottleNeck_dim, kernel_size, stride=stride,
                                padding=padding, groups=bottleNeck_dim, bias=bias)
        self.dconv2 = nn.BatchNorm2d(bottleNeck_dim)
        self.dconv3 = nn.ReLU6()

        # Squeeze and Excitation phase
        se_dim = int(se_ratio * bottleNeck_dim)
        self.se1 = nn.Conv2d(bottleNeck_dim, se_dim, 1, stride=1, bias=bias)
        self.se2 = nn.ReLU6()
        self.se3 = nn.Conv2d(se_dim, bottleNeck_dim, 1, stride=1, bias=bias)
        self.se4 = nn.Sigmoid()

        # Output Convolution
        self.out1 = nn.Conv2d(bottleNeck_dim, output_dim, 1, stride=1, bias=bias)
        self.out2 = nn.BatchNorm2d(output_dim)
        self.out3 = nn.Dropout2d(dropout)

    def forward(self, inputs):
        if self.expansion_ratio != 1:
            x = self.exp1(inputs)
            x = self.exp2(x)
            x = self.exp3(x)
        else:
            x = inputs

        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)

        # Global pooling
        se_tensors = torch.mean(x, dim=(2, 3)).view(x.shape[0], x.shape[1], 1, 1)
        se_tensors = self.se1(se_tensors)
        se_tensors = self.se2(se_tensors)
        se_tensors = self.se3(se_tensors)
        se_tensors = self.se4(se_tensors)
        x = x * se_tensors

        x = self.out1(x)
        x = self.out2(x)
        x = self.out3(x)

        if(self.stride == 1 and self.input_dim == self.output_dim):
            x += inputs

        return x


# In[4]:

# # EfficientNet

# In[4]:


class EfficientNet(nn.Module):
    def __init__(self, width, depth, input_dim,
                 dropout=0.2,
                 classes=3,
                 arg_blocks=arg_blocks):
        super(EfficientNet, self).__init__()
        # stage 1
        output_dim = int(round(arg_blocks[0]['input_dim'] * width))
        self.stage1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, 3,
                                              stride=2,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(output_dim),
                                    nn.ReLU6())

        # stage 2~8
        conv_list = []
        for block in arg_blocks:
            input_dim = int(round(block['input_dim'] * width))
            output_dim = int(round(block['output_dim'] * width))
            repeat_times = int(round(block['repeat_times'] * depth, 0)) - 1
            conv_list.append(MBConv(input_dim,
                                    output_dim,
                                    block['kernel_size'],
                                    block['stride'],
                                    block['padding'],
                                    block['expansion_ratio'],
                                    block['se_ratio'],
                                    block['bias'],
                                    block['dropout']))
            for repeat in range(repeat_times):
                conv_list.append(MBConv(output_dim,
                                        output_dim,
                                        block['kernel_size'],
                                        1,
                                        block['padding'],
                                        block['expansion_ratio'],
                                        block['se_ratio'],
                                        block['bias'],
                                        block['dropout']))
        self.stage28 = nn.Sequential(*conv_list)

        # stage 9~10
        self.stage9 = nn.Sequential(nn.Conv2d(output_dim, int(round(1280 * width)), 1,
                                              stride=1,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(int(round(1280 * width))),
                                    nn.Dropout2d(dropout),
                                    nn.ReLU6())
        self.stage10 = nn.Linear(int(round(1280 * width)), classes)
        
    def weights_init_uniform_rule(self, m):
        classname = m.__class__.__name__
        if (classname == 'Conv2d'):
            n = torch.tensor(m.in_channels, dtype=torch.float)
            y = 1.0/torch.sqrt(n)
            m.weight.data.uniform_(-y, y)
        if (classname == 'Linear'):
            n = torch.tensor(m.in_features, dtype=torch.float)
            y = 1.0/torch.sqrt(n)
            m.weight.data.uniform_(-y, y)
            
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage28(x)
        x = self.stage9(x)
        x = self.stage10(x.mean(dim=(2, 3)))
        return x
