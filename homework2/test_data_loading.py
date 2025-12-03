"""
Test script to verify data loading functionality
Run this to ensure the dataset is properly accessible
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES']='0'
class ThermalDataset(Dataset):
    """Custom Dataset for thermal images"""
    
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'non_icas': 0, 'icas': 1}
        
        # Load all image paths and labels
        for class_name in ['non_icas', 'icas']:
            class_path = os.path.join(data_path, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def test_data_loading():
    """Test the data loading functionality"""
    
    data_path = './dataset/ICAS'
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset path {data_path} does not exist!")
        return False
    
    # Basic transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create dataset
    dataset = ThermalDataset(data_path, transform=transform)
    
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(dataset)}")
    
    # Count samples per class
    class_counts = {'non_icas': 0, 'icas': 0}
    for _, label in dataset.samples:
        if label == 0:
            class_counts['non_icas'] += 1
        else:
            class_counts['icas'] += 1
    
    print(f"Class distribution:")
    print(f"  Non-ICAS: {class_counts['non_icas']} samples")
    print(f"  ICAS: {class_counts['icas']} samples")
    
    # Test data loader
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Get a sample batch
    sample_batch = next(iter(data_loader))
    images, labels = sample_batch
    
    print(f"\nSample batch:")
    print(f"  Batch size: {images.shape[0]}")
    print(f"  Image shape: {images.shape}")
    print(f"  Labels: {labels}")
    
    # Visualize sample images
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    class_names = ['Non-ICAS', 'ICAS']
    
    for i in range(4):
        # Convert tensor to numpy and transpose for matplotlib
        img = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f'{class_names[labels[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nData loading test completed successfully!")
    print(f"Sample images saved as 'sample_images.png'")
    
    return True

if __name__ == "__main__":
    test_data_loading()
