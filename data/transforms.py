"""
Data transforms for CORe50 continual object recognition.
Follows standard PyTorch / ImageNet conventions for 128x128 resolution.
"""

import torchvision.transforms as transforms


def get_transforms(image_size: int = 128, is_train: bool = True):
    """
    Returns train or evaluation torchvision transforms.
    
    Args:
        image_size: Target image dimension (CORe50 native is 128).
        is_train: Whether to apply data augmentation for training.
    """
    # ImageNet standard normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def denormalize_tensor(tensor):
    """
    Utility to convert normalized tensor back to [0, 1] RGB for visualization.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    t = tensor.clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    return t.clamp(0, 1)
