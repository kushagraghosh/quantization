import torch
import numpy as np
from finetune import FineTunedResNet18
from smoothquant import SmoothQuantFineTunedResNet18

def compare_models():
    # Load original finetuned model
    finetuned_model = FineTunedResNet18(num_classes=29)
    finetuned_state = torch.load('resnet18_finetuned.pt')
    finetuned_model.load_state_dict(finetuned_state)

    # Load quantized model
    quantized_model = SmoothQuantFineTunedResNet18(num_classes=29)
    quantized_state = torch.load('resnet18_quantized.pt')
    quantized_model.load_state_dict(quantized_state['model_state_dict'])

    # Compare weights and print statistics
    print("Comparing models:")
    for (name1, param1), (name2, param2) in zip(finetuned_model.named_parameters(), 
                                               quantized_model.named_parameters()):
        if param1.data.shape == param2.data.shape:
            diff = torch.abs(param1.data - param2.data)
            print(f"\nLayer: {name1}")
            print(f"Max difference: {torch.max(diff).item():.6f}")
            print(f"Mean difference: {torch.mean(diff).item():.6f}")
            print(f"Data type: {param1.data.dtype} vs {param2.data.dtype}")
            print(f"Memory size: {param1.data.element_size() * param1.data.nelement()} vs "
                  f"{param2.data.element_size() * param2.data.nelement()} bytes")

    # Print model sizes
    def get_model_size(model_dict):
        total_size = 0
        for param in model_dict.values():
            total_size += param.element_size() * param.nelement()
        return total_size / (1024 * 1024)  # Convert to MB

    finetuned_size = get_model_size(finetuned_model.state_dict())
    quantized_size = get_model_size(quantized_model.state_dict())

    print(f"\nTotal model sizes:")
    print(f"Finetuned model: {finetuned_size:.2f} MB")
    print(f"Quantized model: {quantized_size:.2f} MB")

if __name__ == "__main__":
    compare_models()