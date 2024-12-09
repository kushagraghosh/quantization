import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from finetune import FineTunedResNet18, CustomDataset


def quant(x, scale, zero, maxq):
    # Map the values in x (input) to finite set of possible vals
    if maxq < 0:
        #Binary: if val closer to scale, return scale. if closer to zero, return zero
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero 
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq) # vals b/w [0, maxq]
    return scale * (q - zero) 

def find_params(x, bits=8):
    maxq = 2 ** bits - 1  # Maximum quantization val
    xmax = x.max()  # Maximum value in the tensor
    xmin = x.min()  # Minimum value in the tensor
    scale = (xmax - xmin) / maxq  # Compute the scale factor
    zero = torch.round(-xmin / scale)  # Compute the zero-point
    return scale, zero, maxq

def gptq_quantization(model, blocksize=2, bits=8):
    for name, param in model.named_parameters():  # Iterate through all model parameters
        if param.requires_grad and len(param.shape) > 1:  # Only process weight matrices
            with torch.no_grad():  # Ensure no gradients are computed during quantization
                W = param.data  # Access the weight matrix
                print(f"Quantizing layer: {name}, shape: {W.shape}")

                # Handle convolutional layer weights
                orig_shape = W.shape
                if len(orig_shape) == 4:  # Convolutional layers
                    W = W.reshape(W.shape[0], -1)  # Reshape to 2D matrix
                drow, dcol = W.shape  # Get dimensions of the weight matrix
                Q = torch.zeros_like(W)  # Initialize matrix to store quantized weights
                H_inv = torch.inverse(W.T @ W)  # Approximate inverse Hessian matrix
                
                for i in range(0, dcol, blocksize):  # Process columns in blocks
                    block_end = min(i + blocksize, dcol)  # Determine the end of the block
                    E = torch.zeros((drow, blocksize))  # Initialize block quantization errors

                    # Iterate through each column in the current block
                    for j in range(i, block_end):
                        scale, zero, maxq = find_params(W[:, j], bits)  # Find quantization parameters
                        Q[:, j] = quant(W[:, j], scale, zero, maxq)  # Quantize the column
                        E[:, j - i] = (W[:, j] - Q[:, j]) / H_inv[j, j]  # Compute quantization error
                        W[:, j:block_end] -= torch.outer(E[:, j - i], H_inv[j, j:block_end])  # Update weights in block
                    
                    # Update remaining weights outside the block
                    if block_end < dcol:
                        W[:, block_end:] -= E @ H_inv[i:block_end, block_end:]

                # Handle convolutional layer weights
                if len(orig_shape) == 4:  # Convolutional layers
                    Q = Q.reshape(orig_shape)
                param.data = Q  # Replace original weights with quantized weights
    return model  # Return the quantized model


class GPTQFineTunedResNet18(FineTunedResNet18):
    def __init__(self, num_classes):
        super(GPTQFineTunedResNet18, self).__init__(num_classes)

def gptq_quantization_resnet18():
    # Device configuration (quantization requires CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = GPTQFineTunedResNet18(num_classes=29)
    
    # Load pre-trained weights
    state_dict = torch.load('resnet18_finetuned.pt', map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully")
    
    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Quantize the model
    print("Quantizing model...")
    model = gptq_quantization(model, blocksize=2, bits=8)
    
    # Test the model
    print("Testing quantized model...")
    test_dataset = CustomDataset(
        images_dir='Dataset/images/test',
        labels_dir='Dataset/labels/test',
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )
    
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy on test set: {100 * correct / total}%')
    
    # Save the quantized model
    torch.save(model.state_dict(), 'resnet18_gptquantized.pt')
    print("Model saved as resnet18_gptquantized.pt")

if __name__ == "__main__":
    #gptq_quantization_resnet18()
    #load resnet18_gptquantized and run it on validation set
    device = torch.device('cpu')
    print(f"Using device: {device}")

    model = GPTQFineTunedResNet18(num_classes=29)
    state_dict = torch.load('resnet18_gptquantized.pt', map_location=device)
    model.to(device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully")

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Validation dataset
    val_dataset = CustomDataset(
        images_dir='Dataset/images/val',
        labels_dir='Dataset/labels/val',
        transform=transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    # Test the model
    print("Testing quantized model...")
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")