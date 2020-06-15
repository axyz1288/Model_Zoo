#!/usr/bin/env python
# coding: utf-8

import torch
import torch.optim as optim

def norm(x):
    return (x - x.min()) / (x.max() - x.min())

def add_gaussian_noise(x, mean, std):
    return x + mean + torch.randn(x.size()) * std

def get_saliency(x, label, model):
    (model.cuda()).eval()
    x.requires_grad_(True)

    predict = model(x.cuda())
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(predict, label.cuda())
    loss.backward()

    # absolute gradient
    saliencies = x.grad.abs().permute(0, 2, 3, 1).detach().cpu()
    saliencies = torch.stack([norm(img_grad) for img_grad in saliencies])
    return saliencies 


def fisher_sensitivity(x, label, model, iteration=1, lr=0.001):
    model.cuda().eval()
    loss_func = torch.nn.CrossEntropyLoss()
    x1 = add_gaussian_noise(x, 0.5, 0.1)
    x1.requires_grad_(True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    optimizer.zero_grad()
    predict = model(x1.cuda())
    loss = loss_func(predict, label.cuda())
    loss.backward()
    optimizer.step()
    
    gradient = x1.grad
    d_gradient = 0.0
    fisher_sensitivity = 0.0
    for iter in range(iteration):
        x1 = add_gaussian_noise(x, 0.5, 0.1)
        x1.requires_grad_(True)
        
        optimizer.zero_grad()
        predict = model(x1.cuda())
        loss = loss_func(predict, label.cuda())
        loss.backward()
        optimizer.step()
        
        d_gradient = (x1.grad - gradient) / x.mean()
        fisher_sensitivity += d_gradient
        gradient = x1.grad
        
    fisher_sensitivity = fisher_sensitivity.permute(0, 2, 3, 1).detach().cpu()
    return norm(fisher_sensitivity)
    
layer_activations = None
def filter_explaination(x, model, layer, iteration=100, lr=1):
    # layer: 想要指定第幾層 layer 
    model.cuda().eval()
       
    # hook 指定的 layer
    def hook(model, input, output):
        global layer_activations
        layer_activations = output
    hook_handler = layer.register_forward_hook(hook)
    
    """
    Filter activation: 我們先觀察 x 經過被指定 layer 的 activation map
    """
    model(x.cuda())
    filter_activations = layer_activations.detach().cpu()  
    
    """
    Filter visualization: 接著我們要找出可以最大程度 activate 該 filter 的圖片
    """
    x1 = add_gaussian_noise(x, 0.5, 0.1)
    x1.requires_grad_(True)
    # 我們要對 input image 算偏微分
    optimizer = optim.Adam([x1], lr=lr)
    # 利用偏微分和 optimizer，逐步修改 input image 來讓 filter activation 越來越大
    for iter in range(iteration):
        optimizer.zero_grad()
        model(x1.cuda())
        
        objective = -layer_activations.sum()
        # 與上一個作業不同的是，我們並不想知道 image 的微量變化會怎樣影響 final loss
        # 我們想知道的是，image 的微量變化會怎樣影響 activation 的程度
        # 因此 objective 是 filter activation 的加總，然後加負號代表我們想要做 maximization

        objective.backward()
        # 計算 filter activation 對 input image 的偏微分
        optimizer.step()
        # 修改 input image 來最大化 filter activation
    
    filter_visualization = x1.permute(0, 2, 3, 1).detach().cpu()
    # 完成圖片修改，只剩下要畫出來，因此可以直接 detach 並轉成 cpu tensor

    hook_handler.remove()
    # 很重要：一旦對 model register hook，該 hook 就一直存在。如果之後繼續 register 更多 hook
    # 那 model 一次 forward 要做的事情就越來越多，甚至其行為模式會超出你預期 (因為你忘記哪邊有用不到的 hook 了)
    # 因此事情做完了之後，就把這個 hook 拿掉，下次想要再做事時再 register 就好了。

    return filter_activations, filter_visualization