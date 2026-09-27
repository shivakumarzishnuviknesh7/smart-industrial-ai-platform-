import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Simple CNN for FashionMNIST classification.

    Input:
        [batch_size, 1, 28, 28]

    Output:
        [batch_size, num_classes]
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64 * 7 * 7,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(config):
    """
    Create model from YAML configuration.
    """

    model_name = config["model"]["name"].lower()
    num_classes = config["model"]["num_classes"]

    if model_name == "baseline_cnn":
        model = BaselineCNN(
            num_classes=num_classes
        )
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            "Currently supported: baseline_cnn"
        )

    return model