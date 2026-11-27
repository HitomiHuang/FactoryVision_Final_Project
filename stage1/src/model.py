"""Build the from-scratch CNN for the Stage 1 ablation settings."""

from torch import nn


MODEL_NAME = "SimpleCNN"
CHANNELS = (32, 64, 128, 256)


def conv_block(in_channels, out_channels):
    """Return one Conv3x3 -> ReLU -> MaxPool block that halves the image size."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class SimpleCNN(nn.Module):
    """Four conv blocks, global average pooling and one linear classifier."""

    def __init__(self, in_channels=1, num_classes=6, dropout=0.0):
        """Use Dropout before the classifier only when dropout > 0 (method B)."""
        super().__init__()
        blocks = []
        previous = in_channels
        for width in CHANNELS:
            blocks.append(conv_block(previous, width))
            previous = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(previous, num_classes)

    def forward(self, x):
        """Return one logit per class for each image in the batch."""
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(self.dropout(x))


def build_model(in_channels=1, num_classes=6, dropout=0.0):
    """Create a randomly initialised SimpleCNN (no pretrained weights)."""
    return SimpleCNN(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
