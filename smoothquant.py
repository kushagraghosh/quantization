import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from finetune import FineTunedResNet18, CustomDataset

def smooth_quant_cnn_per_batch(X, W, alpha=0.5):
    max_X = torch.amax(torch.abs(X), dim=(2, 3), keepdim=False)
    max_W = torch.amax(torch.abs(W), dim=(0, 2, 3), keepdim=False)
    s = torch.pow(max_X, alpha) / torch.pow(max_W, 1 - alpha)
    X_hat = X / s[:, :, None, None]
    W_hat = W * s.mean(dim=0).view(1, -1, 1, 1)
    return X_hat, W_hat

class SmoothQuantFineTunedResNet18(FineTunedResNet18):
    def __init__(self, num_classes):
        super(SmoothQuantFineTunedResNet18, self).__init__(num_classes)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()
        
    def _make_layer_quantized(self, x, layer):
        identity = x
        
        # First conv block
        out = layer.conv1(x)
        out = layer.bn1(out)
        out = layer.relu(out)
        
        # Second conv block
        out = layer.conv2(out)
        out = layer.bn2(out)
        
        # Handle downsample
        if hasattr(layer, 'downsample') and layer.downsample is not None:
            identity = layer.downsample(x)
            
        # Quantized addition
        out = self.skip_add.add(out, identity)
        out = layer.relu(out)
        return out

    def forward(self, x):
        # Initial quantization
        x = self.quant(x)
        
        # Initial conv + pooling
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        
        # Layer blocks with quantized residual connections
        for block in self.resnet18.layer1:
            x = self._make_layer_quantized(x, block)
        for block in self.resnet18.layer2:
            x = self._make_layer_quantized(x, block)
        for block in self.resnet18.layer3:
            x = self._make_layer_quantized(x, block)
        for block in self.resnet18.layer4:
            x = self._make_layer_quantized(x, block)
        
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet18.fc(x)
        
        # Final dequantization
        x = self.dequant(x)
        return x

def prepare_model_for_quantization(model, calibration_loader):
    model.eval()
    
    # Configure quantization parameters
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.backends.quantized.engine = 'fbgemm'
    
    # Fuse layers for better quantization
    torch.quantization.fuse_modules(
        model.resnet18,
        [['conv1', 'bn1', 'relu']],
        inplace=True
    )
    
    # Prepare for quantization
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate with data
    with torch.no_grad():
        for i, (data, _) in enumerate(calibration_loader):
            model(data)
            if i >= 100:  # Use more calibration batches
                break
    
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    return model

def main():
    # Device configuration (quantization requires CPU)
    device = torch.device('cpu')
    
    # Initialize model
    model = SmoothQuantFineTunedResNet18(num_classes=29)
    
    # Load pre-trained weights
    state_dict = torch.load('resnet18_finetuned.pt', map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create calibration dataset and loader
    cal_dataset = CustomDataset(
        images_dir='Dataset/images/val',
        labels_dir='Dataset/labels/val',
        transform=transform
    )
    calibration_loader = DataLoader(
        cal_dataset,
        batch_size=32,
        shuffle=False
    )
    
    # Prepare and quantize model
    model = prepare_model_for_quantization(model, calibration_loader)
    
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
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': 100 * correct / total,
    }, 'resnet18_quantized.pt')
    print("Model saved as resnet18_smoothquantized.pt")

if __name__ == "__main__":
    main()