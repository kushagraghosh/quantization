import copy
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from finetune import CustomDataset
import math

class FineTunedResNet18(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet18, self).__init__()
        # Load pre-trained ResNet18 model
        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Modify the fully connected layer to match the number of classes
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        # Define the forward pass
        return self.resnet18(x)

class GPTQFineTunedResNet18(FineTunedResNet18):
    def __init__(self, num_classes):
        super(GPTQFineTunedResNet18, self).__init__(num_classes)

def quant(x, scale, zero, maxq):
    # Map the values in x (input) to finite set of possible vals
    if maxq < 0:
        #Binary: if val closer to scale, return scale. if closer to zero, return zero
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero 
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq) # vals b/w [0, maxq]
    return scale * (q - zero) 

class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(self, bits):
        self.maxq = torch.tensor(2 ** bits - 1) #max value based on bits

    def find_params(self, x, weight=False):
        shape = x.shape

        x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0])
        xmin = torch.minimum(x.min(1)[0], tmp) #max val in first dim (row)
        xmax = torch.maximum(x.max(1)[0], tmp) #min val in first dim

        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq # 
            self.zero = torch.round(-xmin / self.scale)

        if weight: 
            tmp = shape[0] #repeat scale and zero across first dimension
        else:
            tmp = shape[1] if len(shape) != 3 else shape[2] #repeat scale and zero across second or third dimension
        self.scale = self.scale.repeat(tmp)
        self.zero = self.zero.repeat(tmp)

        #ensures that the scale and zero point have the correct shape for input tensor during quantization
        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)
    
    def quantize(self, x):
        #calls our quantize function
        if self.ready():
            return quant(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

class GPTQ:
    def __init__(self, layer): 
        #quantize layer by layer
        self.quantizer = None
        self.layer = layer
        W = layer.weight.data.clone().float()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1) #flatten layer to be 2d matrix
        self.rows, self.columns = W.shape
        self.H = torch.zeros((self.columns, self.columns)) #Hessian matrix of the objective fucnction with respect to the weights
        self.nsamples = 0 #keeps track of num samples used in quant process

    def add_input_batch(self, input):
        print(input.shape)
        #inp and out are tensors representing input and output of the layer
        if len(input.shape) == 2:
            input = input.unsqueeze(0) #add a new dimension to make the single input into a batch size of 1 
        first_dim_of_input = input.shape[0] #useful when updating hessian
        if isinstance(self.layer, nn.Linear):
            if len(input.shape) == 3:
                input = input.reshape((-1, input.shape[-1])) # (feature size, batch size)
            input = input.t() #tranpose input
        if isinstance(self.layer, nn.Conv2d):
            # trying to extract the patches from input tensor corresponding to the kernel in the conv layer
            # unfolds the input tensor into a series of smaller, overlapping patches
            # determines how the kernel is applied across the input tensor
            #https://stackoverflow.com/questions/53972159/how-does-pytorchs-fold-and-unfold-work
            input = nn.Unfold(self.layer.kernel_size, dilation=self.layer.dilation,
                            padding=self.layer.padding, stride=self.layer.stride)(input)
            #output is (batch_size, num_patches, patch_size)
            input = input.permute(1, 2, 0).flatten(1) #(a patch, an element within the patch)
        self.H *= self.nsamples / (self.nsamples + first_dim_of_input) 
        #update Hessian matrix by multiplying with a ratio of the old number of samples
        # gives more weight to the existing Hessian matrix while incorporating new samples
        self.nsamples += first_dim_of_input
        input = (math.sqrt(2 / self.nsamples) * input.float())
        print(input.shape)
        self.H += input.matmul(input.t()) #update Hessian matrix with the new info from a new batch of inputs

    def quantize_weights(self, blocksize=128, percent_damp=0.01):
        #blocksize is 128 in the original paper so apply gptq algo to 128 columns at a time while keeping updates contained to those cols
        W = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1) #Maintain 2D matrix of weights to apply quant correctly

        self.quantizer.find_params(W, weight=True)

        H = self.H #Hessian matrix determines how much to adjust each weight to minimize quantization error
        del self.H
        #avoid division by 0 by replacing any 0's in hessian matrix with 1
        zero_entries = torch.diag(H) == 0
        H[zero_entries, zero_entries] = 1
        W[:, zero_entries] = 0

        loss_of_quant_weights = torch.zeros_like(W)
        Q = torch.zeros_like(W) #quantized weights

        damp = percent_damp * torch.mean(torch.diag(H)) #percentage of the mean of the diagonal of the Hessian matrix
        diag = torch.arange(self.columns)
        H[diag, diag] += damp #add damping to the diagonal of the Hessian matrix
        
        #cholesky is used to compute all info for Hessian matrix upfront
        H = torch.linalg.cholesky(H) #hessian matrix factored with cholesky decomposition
        H = torch.cholesky_inverse(H) 
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H #hessian of the upper triangular matrix

        #loop over the columns of the weight matrix for each layer
        #each iteration processes a block of columns at a time
        for col_index in range(0, self.columns, blocksize): 
            end_col_index_of_block = min(col_index + blocksize, self.columns)
            num_cols_in_block = end_col_index_of_block - col_index
            #initializations
            block_weights = W[:, col_index:end_col_index_of_block].clone()
            quantized_block_weights = torch.zeros_like(block_weights)
            block_error = torch.zeros_like(block_weights)
            block_loss_of_weights = torch.zeros_like(block_weights)
            block_hessian_submatrix = Hinv[col_index:end_col_index_of_block, col_index:end_col_index_of_block]

            for col in range(num_cols_in_block):
                w = block_weights[:, col]
                d = block_hessian_submatrix[col, col] #inverse of submatrix that corresponds to the current column

                q = quant(
                    w.unsqueeze(1), 
                    self.quantizer.scale, 
                    self.quantizer.zero, 
                    self.quantizer.maxq
                ).flatten()
                quantized_block_weights[:, col] = q
                block_loss_of_weights[:, col] = (w - q) ** 2 / d ** 2 #from paper: calc loss and then update remaining weights based on hessian inverse matrix

                col_error = (w - q) / d
                block_weights[:, col:] -= col_error.unsqueeze(1).matmul(block_hessian_submatrix[col, col:].unsqueeze(0)) #remaining cols in block are updated by subtracting the error and corresponding rows in the hessian submatrix
                block_error[:, col] = col_error

            Q[:, col_index:end_col_index_of_block] = quantized_block_weights
            loss_of_quant_weights[:, col_index:end_col_index_of_block] = block_loss_of_weights / 2

            W[:, end_col_index_of_block:] -= block_error.matmul(Hinv[col_index:end_col_index_of_block, end_col_index_of_block:])

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.dtype) #self.layer assigned quantized layer in the right self.layer.weight.shape
    def free(self):
        self.H = None
        torch.cuda.empty_cache()

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

if __name__ == "__main__":
    wbits = 4

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(images_dir='Dataset/images/train', labels_dir='Dataset/labels/train', transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = CustomDataset(images_dir='Dataset/images/test', labels_dir='Dataset/labels/test', transform=transform)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    wquant = wbits < 32

    modelp = GPTQFineTunedResNet18(num_classes=29)
    state_dict = torch.load('resnet18_finetuned.pt', map_location='cpu')
    modelp.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully")
    layersp = find_layers(modelp)

    modeld = GPTQFineTunedResNet18(num_classes=29)
    state_dict = torch.load('resnet18_finetuned.pt', map_location='cpu')
    modeld.load_state_dict(state_dict, strict=False)
    layersd = find_layers(modeld)

    gptq = {}
    for name in layersp:
        layer = layersp[name]
        gptq[name] = GPTQ(layer)
        if wquant:
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(wbits)

    cache = {}
    def add_batch(name):
        def tmp(layer, inp):
            gptq[name].add_input_batch(inp[0].data)
        return tmp
    handles = []
    for name in gptq:
        handles.append(layersd[name].register_forward_hook(add_batch(name)))
    for h in handles:
        h.remove()
    for name in gptq:
        print(name)
        print('Quantizing ...')
        gptq[name].quantize_weights()
        gptq[name].free()
        
    torch.save(modelp.state_dict(), 'resnet18_gptquantized.pt')
