import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=4, pretrained=True, dropout=0.3):
    """
    Builds EfficientNet-B0 with dropout for multilabel classification.
    EfficientNet-B0 is faster and more efficient than ResNet-50.
    
    Args:
        num_classes (int): Number of attributes (4)
        pretrained (bool): Use ImageNet pretrained weights
        dropout (float): Dropout rate to prevent overfitting
    
    Returns:
        model: EfficientNet-B0 with custom classifier head
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    
    # Freeze backbone initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier with dropout
    # EfficientNet-B0: 1280 features → 4 classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def unfreeze_model(model, freeze_layers=5):
    """
    Gradually unfreezes model layers for fine-tuning.
    Keeps early layers frozen to prevent overfitting.
    
    Args:
        model: The model to unfreeze
        freeze_layers (int): Number of early feature blocks to keep frozen
    """
    # Unfreeze all parameters first
    for param in model.parameters():
        param.requires_grad = True
    
    # Re-freeze early layers (they learn generic features)
    for i, block in enumerate(model.features):
        if i < freeze_layers:
            for param in block.parameters():
                param.requires_grad = False
    
    return model


def get_pos_weights(device):
    """
    Class imbalance weights for BCEWithLogitsLoss.
    Computed from dataset distribution analysis.
    """
    weights = torch.tensor([0.13, 0.24, 0.89, 11.96], dtype=torch.float32)
    return weights.to(device)