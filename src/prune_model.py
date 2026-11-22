import os
import time
from statistics import mean

import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

REGISTERED_MODEL_NAME = "industrial-baseline-cnn"
MODEL_ALIAS = "champion"

MODEL_URI = (
    f"models:/{REGISTERED_MODEL_NAME}"
    f"@{MODEL_ALIAS}"
)

PRUNING_AMOUNT = 0.30

BATCH_SIZE = 64
NUM_WARMUP_BATCHES = 5
NUM_BENCHMARK_BATCHES = 50

OUTPUT_PATH = "artifacts/pruned_model.pth"


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

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
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

def evaluate_accuracy(
    model,
    test_loader,
    device,
):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

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

    return correct / total


# ---------------------------------------------------------
# Synchronization
# ---------------------------------------------------------

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------
# Benchmark
# ---------------------------------------------------------

def benchmark_inference(
    model,
    test_loader,
    device,
):
    model.eval()

    iterator = iter(test_loader)

    # Warm-up
    with torch.no_grad():

        for _ in range(
            NUM_WARMUP_BATCHES
        ):

            try:
                images, _ = next(iterator)

            except StopIteration:
                iterator = iter(test_loader)
                images, _ = next(iterator)

            images = images.to(device)

            _ = model(images)

            synchronize(device)

    # Benchmark
    iterator = iter(test_loader)

    batch_latencies = []
    total_images = 0

    with torch.no_grad():

        for _ in range(
            NUM_BENCHMARK_BATCHES
        ):

            try:
                images, _ = next(iterator)

            except StopIteration:
                break

            images = images.to(device)

            synchronize(device)

            start = time.perf_counter()

            _ = model(images)

            synchronize(device)

            end = time.perf_counter()

            latency = end - start

            batch_latencies.append(
                latency
            )

            total_images += images.size(0)

    total_time = sum(
        batch_latencies
    )

    avg_batch_latency = mean(
        batch_latencies
    )

    avg_image_latency = (
        total_time / total_images
    )

    throughput = (
        total_images / total_time
    )

    return (
        avg_batch_latency,
        avg_image_latency,
        throughput,
    )


# ---------------------------------------------------------
# Parameter count
# ---------------------------------------------------------

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# ---------------------------------------------------------
# Model size
# ---------------------------------------------------------

def get_model_size_mb(
    model,
    path,
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        path,
    )

    size_bytes = os.path.getsize(
        path
    )

    return size_bytes / (
        1024 * 1024
    )


# ---------------------------------------------------------
# Sparsity
# ---------------------------------------------------------

def calculate_sparsity(model):
    zero_weights = 0
    total_weights = 0

    for module in model.modules():

        if isinstance(
            module,
            (nn.Conv2d, nn.Linear),
        ):

            weights = module.weight

            zero_weights += (
                weights == 0
            ).sum().item()

            total_weights += (
                weights.numel()
            )

    if total_weights == 0:
        return 0.0

    return (
        zero_weights / total_weights
    )


# ---------------------------------------------------------
# Apply pruning
# ---------------------------------------------------------

def apply_pruning(model):
    print(
        f"\nApplying "
        f"{PRUNING_AMOUNT * 100:.0f}% "
        f"L1 unstructured pruning..."
    )

    for module_name, module in (
        model.named_modules()
    ):

        if isinstance(
            module,
            (nn.Conv2d, nn.Linear),
        ):

            print(
                f"Pruning layer: "
                f"{module_name}"
            )

            prune.l1_unstructured(
                module,
                name="weight",
                amount=PRUNING_AMOUNT,
            )

            # Permanently apply the mask.
            #
            # Without prune.remove(), PyTorch keeps:
            # weight_orig + weight_mask.
            #
            # After removal, the actual weight tensor
            # contains the zeros.
            prune.remove(
                module,
                "weight",
            )

    return model


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print(
        "\n=== Model Pruning Experiment ===\n"
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    print(
        f"Loading champion model: "
        f"{MODEL_URI}"
    )

    model = mlflow.pytorch.load_model(
        MODEL_URI
    )

    model = model.to(device)
    model.eval()

    print(
        "Model loaded successfully."
    )

    test_loader = get_test_loader()

    total_parameters = count_parameters(
        model
    )

    # -----------------------------------------------------
    # Baseline
    # -----------------------------------------------------

    print(
        "\nEvaluating baseline model..."
    )

    baseline_accuracy = evaluate_accuracy(
        model,
        test_loader,
        device,
    )

    (
        baseline_batch_latency,
        baseline_image_latency,
        baseline_throughput,
    ) = benchmark_inference(
        model,
        test_loader,
        device,
    )

    baseline_temp_path = (
        "artifacts/"
        "_baseline_pruning_test.pth"
    )

    baseline_size = get_model_size_mb(
        model,
        baseline_temp_path,
    )

    os.remove(
        baseline_temp_path
    )

    baseline_sparsity = (
        calculate_sparsity(model)
    )

    # -----------------------------------------------------
    # Pruning
    # -----------------------------------------------------

    model = apply_pruning(
        model
    )

    pruned_sparsity = (
        calculate_sparsity(model)
    )

    # -----------------------------------------------------
    # Evaluate pruned model
    # -----------------------------------------------------

    print(
        "\nEvaluating pruned model..."
    )

    pruned_accuracy = evaluate_accuracy(
        model,
        test_loader,
        device,
    )

    (
        pruned_batch_latency,
        pruned_image_latency,
        pruned_throughput,
    ) = benchmark_inference(
        model,
        test_loader,
        device,
    )

    pruned_size = get_model_size_mb(
        model,
        OUTPUT_PATH,
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    accuracy_drop = (
        baseline_accuracy
        - pruned_accuracy
    )

    size_change = (
        baseline_size
        - pruned_size
    )

    print(
        "\n================================"
    )

    print(
        "       PRUNING RESULTS"
    )

    print(
        "================================"
    )

    print(
        f"Model parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Pruning amount: "
        f"{PRUNING_AMOUNT * 100:.0f}%"
    )

    print(
        "\n--- Baseline ---"
    )

    print(
        f"Accuracy: "
        f"{baseline_accuracy * 100:.2f}%"
    )

    print(
        f"Sparsity: "
        f"{baseline_sparsity * 100:.2f}%"
    )

    print(
        f"Model size: "
        f"{baseline_size:.3f} MB"
    )

    print(
        f"Batch latency: "
        f"{baseline_batch_latency * 1000:.3f} ms"
    )

    print(
        f"Image latency: "
        f"{baseline_image_latency * 1000:.3f} ms"
    )

    print(
        f"Throughput: "
        f"{baseline_throughput:.2f} "
        f"images/sec"
    )

    print(
        "\n--- Pruned Model ---"
    )

    print(
        f"Accuracy: "
        f"{pruned_accuracy * 100:.2f}%"
    )

    print(
        f"Sparsity: "
        f"{pruned_sparsity * 100:.2f}%"
    )

    print(
        f"Model size: "
        f"{pruned_size:.3f} MB"
    )

    print(
        f"Batch latency: "
        f"{pruned_batch_latency * 1000:.3f} ms"
    )

    print(
        f"Image latency: "
        f"{pruned_image_latency * 1000:.3f} ms"
    )

    print(
        f"Throughput: "
        f"{pruned_throughput:.2f} "
        f"images/sec"
    )

    print(
        "\n--- Difference ---"
    )

    print(
        f"Accuracy drop: "
        f"{accuracy_drop * 100:.2f} "
        f"percentage points"
    )

    print(
        f"Model size change: "
        f"{size_change:.3f} MB"
    )

    print(
        f"\nSaved pruned model to: "
        f"{OUTPUT_PATH}"
    )

    print(
        "\nPruning experiment complete."
    )


if __name__ == "__main__":
    main()