import os
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

# List of class labels (indices)
labels = ['Road narrows on right', '50 mph speed limit', 'Attention Please-', 'Beware of children', 
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

def analyze_dataset():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Training dataset
    train_dataset = CustomDataset(
        images_dir='Dataset/images/train', #change path as necessary
        labels_dir='Dataset/labels/train',
        transform=transform
    )
    # Validation dataset
    val_dataset = CustomDataset(
        images_dir='Dataset/images/val',
        labels_dir='Dataset/labels/val',
        transform=transform
    )
    # Test dataset
    test_dataset = CustomDataset(
        images_dir='Dataset/images/test',
        labels_dir='Dataset/labels/test',
        transform=transform
    )
    combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    # Count classes
    class_counts = {}
    for _, lbls in dataloader:
        for label in lbls:
            class_name = labels[label.item()]
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

    sorted_class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))
    # plt.bar(sorted_class_counts.keys(), sorted_class_counts.values())
    # plt.xlabel('Classes')
    # plt.ylabel('Frequency')
    # plt.title('Class Distribution in Dataset')
    # plt.xticks(rotation=90, ha='right')
    # plt.tight_layout()  # Adjust layout to include everything
    # plt.show()
    # Plot the 5 most popular classes with an example image and label and count
    top_6_classes = list(sorted_class_counts.keys())[:6]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i, class_name in enumerate(top_6_classes):
        row, col = divmod(i, 3)
        # Find an example image for the class
        for img, lbl in combined_dataset:
            if labels[lbl] == class_name:
                axs[row, col].imshow(img.permute(1, 2, 0))  # Convert from tensor to image format
                axs[row, col].axis('off')
                axs[row, col].set_title(f'Class: {class_name}\nCount: {sorted_class_counts[class_name]}', fontsize=24)
                break

    plt.tight_layout()
    plt.show()
    # save the plot
    # plt.savefig('class_distribution.png')
    plt.savefig('frequent_classes.png')


if __name__ == "__main__":
    analyze_dataset() 