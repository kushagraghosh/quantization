import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import os
from PIL import Image

# List of class labels (indices)
labels = ['-Road narrows on right', '50 mph speed limit', 'Attention Please-', 'Beware of children', 
          'CYCLE ROUTE AHEAD WARNING', 'Dangerous Left Curve Ahead', 'Dangerous Right Curve Ahead', 
          'End of all speed and passing limits', 'Give Way', 'Go Straight or Turn Right', 'Go straight or turn left', 
          'Keep-Left', 'Keep-Right', 'Left Zig Zag Traffic', 'No Entry', 'No_Over_Taking', 
          'Overtaking by trucks is prohibited', 'Pedestrian Crossing', 'Round-About', 'Slippery Road Ahead', 
          'Speed Limit 20 KMPh', 'Speed Limit 30 KMPh', 'Stop_Sign', 'Straight Ahead Only', 'Traffic_signal', 
          'Truck traffic is prohibited', 'Turn left ahead', 'Turn right ahead', 'Uneven Road']

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # Get all image files in the images directory
        self.image_files = sorted(os.listdir(images_dir))  # Sorting to align with labels
        self.label_files = sorted(os.listdir(labels_dir))  # Sorting to align with images

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Convert to RGB if not already

        # Load label
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        with open(label_path, 'r') as f:
            label_idx = int(f.read().strip().split(" ")[0])  # Read label from file

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label_idx

# Define a custom model class to include the forward method
class FineTunedResNet18(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet18, self).__init__()
        # Load pre-trained ResNet18 model
        self.resnet18 = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        # Modify the fully connected layer to match the number of classes
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        # Define the forward pass
        return self.resnet18(x)

# Function to fine-tune the model
def finetune_resnet18(train_dir, val_dir, num_epochs=10, batch_size=32, lr=0.001, device=None):
    # Define the necessary transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),       # Resize images to the size expected by ResNet18
        transforms.ToTensor()               # Convert image to tensor
    ])

    # Define the dataset and dataloaders
    train_dataset = CustomDataset(images_dir='Dataset/images/train', labels_dir='Dataset/labels/train', transform=transform)
    val_dataset = CustomDataset(images_dir='Dataset/images/val', labels_dir='Dataset/labels/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Example of using the DataLoader
    for images, labels in train_loader:
        print(images.shape)  # Tensor shape (batch_size, 3, 224, 224)
        print(labels.shape)  # Tensor shape (batch_size,)
        break  # Just print the first batch to check

    model = FineTunedResNet18(29)

    # Set device for model (default to CUDA if available, else CPU)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Fine-tuning the model
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Validation loop
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

    # Save the fine-tuned model
    model_filename = 'resnet18_finetuned.pt'
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

    return model  # Return the fine-tuned model for further use

# Example usage of the function:
model = finetune_resnet18(train_dir='Dataset/images/train', val_dir='Dataset/images/val', num_epochs=10, batch_size=32, lr=0.001)
