from torch import nn
import torch

en_args_blocks = [
    # layer 1
    {'input_dim': 3, 'output_dim': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1,
      'bias': False, 'dropout': 0.2},
    # layer 2
    {'input_dim': 32, 'output_dim': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1,
      'bias': False, 'dropout': 0.2},
    # layer 3
    {'input_dim': 16, 'output_dim': 24, 'kernel_size': 3, 'stride': 2, 'padding': 1, 
      'bias': False, 'dropout': 0.2},
    # layer 4
    {'input_dim': 24, 'output_dim': 40, 'kernel_size': 5, 'stride': 2, 'padding': 2, 
      'bias': False, 'dropout': 0.2},
    # layer 5
    {'input_dim': 40, 'output_dim': 80, 'kernel_size': 3, 'stride': 2, 'padding': 1, 
      'bias': False, 'dropout': 0.2},
    # layer 6
    {'input_dim': 80, 'output_dim': 112, 'kernel_size': 5, 'stride': 1, 'padding': 2, 
      'bias': False, 'dropout': 0.2},
    # layer 7
    {'input_dim': 112, 'output_dim': 192, 'kernel_size': 5, 'stride': 2, 'padding': 2, 
      'bias': False, 'dropout': 0.2},
    # layer 8
    {'input_dim': 192, 'output_dim': 320, 'kernel_size': 3, 'stride': 1, 'padding': 1, 
      'bias': False, 'dropout': 0.2},
]

de_args_blocks = [
    # layer 1
    {'input_dim': 320, 'output_dim': 192, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'output_padding': 0,
      'bias': False, 'dropout': 0.2},
    # layer 2
    {'input_dim': 192, 'output_dim': 112, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'output_padding': 0,
      'bias': False, 'dropout': 0.2},
    # layer 3
    {'input_dim': 112, 'output_dim': 80, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1,
      'bias': False, 'dropout': 0.2},
    # layer 4
    {'input_dim': 80, 'output_dim': 40, 'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 1,
      'bias': False, 'dropout': 0.2},
    # layer 5
    {'input_dim': 40, 'output_dim': 24, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1,
      'bias': False, 'dropout': 0.2},
    # layer 6
    {'input_dim': 24, 'output_dim': 16, 'kernel_size': 5, 'stride': 1, 'padding': 2, 'output_padding': 0,
      'bias': False, 'dropout': 0.2},
    # layer 7
    {'input_dim': 16, 'output_dim': 32, 'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 1,
      'bias': False, 'dropout': 0.2},
    # layer 8
    {'input_dim': 32, 'output_dim': 3, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'output_padding': 0,
      'bias': False, 'dropout': 0.2},
]

class AutoEncoder(nn.Module):
    def __init__(self, en_args_blocks=en_args_blocks, de_args_blocks=de_args_blocks):
        super(AutoEncoder, self).__init__()
        
        def En_block(input_dim, output_dim, kernel_size, stride, padding, bias, dropout):
            return nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(output_dim),
                nn.ReLU6(),
                nn.Dropout2d(dropout),
            )

        def De_block(input_dim, output_dim, kernel_size, stride, padding, output_padding, bias, dropout):
            return nn.Sequential(
                nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, 
                                    output_padding=output_padding, bias=bias),
                nn.BatchNorm2d(output_dim),
                nn.ReLU6(),
                nn.Dropout2d(dropout),
            )
        
        layers = []
        for args in en_args_blocks:
            layers.append(En_block(args['input_dim'],
                                    args['output_dim'],
                                    args['kernel_size'], 
                                    args['stride'], 
                                    args['padding'], 
                                    args['bias'], 
                                    args['dropout']))
        self.encoder = nn.Sequential(*layers)

        layers = []
        for args in de_args_blocks:
            layers.append(De_block(args['input_dim'],
                                    args['output_dim'],
                                    args['kernel_size'], 
                                    args['stride'], 
                                    args['padding'], 
                                    args['output_padding'],
                                    args['bias'], 
                                    args['dropout']))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x