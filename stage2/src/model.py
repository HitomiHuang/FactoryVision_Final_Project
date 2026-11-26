"""Build the selected pretrained architecture and configure fine-tuning."""

from torch import nn
from torchvision.models import (
    DenseNet121_Weights, ResNet18_Weights, densenet121, resnet18,
)

from data import CLASS_ORDER


# Public ImageNet-1K weights published by torchvision for each architecture.
ARCHITECTURES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "densenet121": (densenet121, DenseNet121_Weights.IMAGENET1K_V1),
}
STRATEGIES = ("feature_extraction", "full_finetune")


def pretrained_weights(architecture):
    """Return the torchvision ImageNet weights definition for one architecture."""
    return ARCHITECTURES[architecture][1]


def build_model(architecture, strategy, pretrained=True):
    """Return a six-class model whose trainable parameters follow the strategy.

    feature_extraction: freeze the whole pretrained backbone, train only the new head.
    full_finetune: train every layer, starting from the pretrained weights.
    Use pretrained=False when a saved state_dict will be loaded afterwards.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    builder, weights = ARCHITECTURES[architecture]
    model = builder(weights=weights if pretrained else None)

    # Replace the 1000-class ImageNet classifier with a new six-class linear head.
    if architecture == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_ORDER))
        head = model.fc
    else:
        model.classifier = nn.Linear(model.classifier.in_features, len(CLASS_ORDER))
        head = model.classifier

    if strategy == "feature_extraction":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head.parameters():
            parameter.requires_grad = True
    return model
