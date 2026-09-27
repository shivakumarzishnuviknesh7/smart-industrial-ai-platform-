from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(
    config: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create training, validation, and test DataLoaders.

    The dataset and training parameters are read from the YAML configuration.

    Returns:
        train_loader
        val_loader
        test_loader
    """

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    dataset_name = config["data"]["dataset"].lower()

    batch_size = config["training"]["batch_size"]

    train_split = config["data"].get("train_split", 0.8)

    seed = config.get("project", {}).get("seed", 42)

    num_workers = config["data"].get("num_workers", 2)

    data_dir = Path(
        config["data"].get(
            "data_dir",
            "./data",
        )
    )

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    generator = torch.Generator().manual_seed(seed)

    # ---------------------------------------------------------
    # Image transformation
    # ---------------------------------------------------------

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5,),
                std=(0.5,),
            ),
        ]
    )

    # ---------------------------------------------------------
    # Dataset selection
    # ---------------------------------------------------------

    if dataset_name == "fashion_mnist":

        full_train_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )

    else:

        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            "Currently supported: fashion_mnist"
        )

    # ---------------------------------------------------------
    # Train / Validation split
    # ---------------------------------------------------------

    train_size = int(
        train_split * len(full_train_dataset)
    )

    val_size = (
        len(full_train_dataset)
        - train_size
    )

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------

    print("Dataset loaded successfully")
    print(f"Dataset: {dataset_name}")
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")
    print(f"Batch size:         {batch_size}")

    return (
        train_loader,
        val_loader,
        test_loader,
    )