import os

import mlflow.pytorch
import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


REGISTERED_MODEL_NAME = "industrial-baseline-cnn"
MODEL_ALIAS = "champion"

MODEL_URI = (
    f"models:/{REGISTERED_MODEL_NAME}"
    f"@{MODEL_ALIAS}"
)

PRUNING_AMOUNT = 0.30
BATCH_SIZE = 64


def load_model():
    print(f"Loading model from: {MODEL_URI}")

    model = mlflow.pytorch.load_model(
        MODEL_URI
    )

    model.eval()

    print("Model loaded successfully")

    return model


def get_test_loader():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,),
                (0.5,),
            ),
        ]
    )

    dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


def evaluate_model(
    model,
    data_loader,
    device,
):
    correct = 0
    total = 0

    model.to(device)
    model.eval()

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    return accuracy


def calculate_sparsity(model):
    total_parameters = 0
    zero_parameters = 0

    for parameter in model.parameters():

        tensor = parameter.detach()

        total_parameters += tensor.numel()

        zero_parameters += (
            tensor == 0
        ).sum().item()

    sparsity = (
        zero_parameters
        / total_parameters
    )

    return sparsity


def apply_pruning(model):
    print(
        f"\nApplying {PRUNING_AMOUNT * 100:.0f}% "
        "L1 unstructured pruning..."
    )

    for module in model.modules():

        if isinstance(
            module,
            (
                torch.nn.Conv2d,
                torch.nn.Linear,
            ),
        ):

            prune.l1_unstructured(
                module,
                name="weight",
                amount=PRUNING_AMOUNT,
            )

            prune.remove(
                module,
                "weight",
            )

    print("Pruning complete")

    return model


def main():

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5001",
    )

    print(
        f"MLflow tracking URI: "
        f"{tracking_uri}"
    )

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model = load_model()

    test_loader = get_test_loader()

    baseline_accuracy = evaluate_model(
        model,
        test_loader,
        device,
    )

    baseline_sparsity = calculate_sparsity(
        model
    )

    print(
        f"\nBaseline accuracy: "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"Baseline sparsity: "
        f"{baseline_sparsity * 100:.2f}%"
    )

    model = apply_pruning(
        model
    )

    pruned_accuracy = evaluate_model(
        model,
        test_loader,
        device,
    )

    pruned_sparsity = calculate_sparsity(
        model
    )

    print(
        f"\nPruned accuracy: "
        f"{pruned_accuracy:.4f}"
    )

    print(
        f"Pruned sparsity: "
        f"{pruned_sparsity * 100:.2f}%"
    )

    accuracy_drop = (
        baseline_accuracy
        - pruned_accuracy
    )

    print(
        f"Accuracy drop: "
        f"{accuracy_drop:.4f}"
    )

    output_path = (
        "artifacts/"
        "pruned_model.pth"
    )

    os.makedirs(
        "artifacts",
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        output_path,
    )

    print(
        f"\nPruned model saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()