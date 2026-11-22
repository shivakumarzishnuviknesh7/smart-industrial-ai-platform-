import os
import time
from statistics import mean

import mlflow.pytorch
import torch
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

BATCH_SIZE = 64
NUM_WARMUP_BATCHES = 5
NUM_BENCHMARK_BATCHES = 50


# ---------------------------------------------------------
# Device selection
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
# Parameter count
# ---------------------------------------------------------

def count_parameters(model):
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# ---------------------------------------------------------
# Model size
# ---------------------------------------------------------

def calculate_model_size(model):
    temp_path = "artifacts/_benchmark_temp_model.pth"

    os.makedirs(
        "artifacts",
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        temp_path,
    )

    size_bytes = os.path.getsize(
        temp_path
    )

    os.remove(
        temp_path
    )

    size_mb = size_bytes / (
        1024 * 1024
    )

    return size_mb


# ---------------------------------------------------------
# Synchronization
# ---------------------------------------------------------

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

def evaluate_accuracy(
    model,
    test_loader,
    device,
):
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(
                0
            )

    accuracy = (
        correct / total
    )

    return accuracy


# ---------------------------------------------------------
# Latency and throughput
# ---------------------------------------------------------

def benchmark_inference(
    model,
    test_loader,
    device,
):
    model.eval()

    iterator = iter(
        test_loader
    )

    # -------------------------------------------------
    # Warm-up
    # -------------------------------------------------

    with torch.no_grad():

        for _ in range(
            NUM_WARMUP_BATCHES
        ):

            try:
                images, _ = next(
                    iterator
                )

            except StopIteration:
                iterator = iter(
                    test_loader
                )

                images, _ = next(
                    iterator
                )

            images = images.to(
                device
            )

            _ = model(
                images
            )

            synchronize(
                device
            )

    # -------------------------------------------------
    # Benchmark
    # -------------------------------------------------

    batch_latencies = []

    total_images = 0

    iterator = iter(
        test_loader
    )

    with torch.no_grad():

        for _ in range(
            NUM_BENCHMARK_BATCHES
        ):

            try:
                images, _ = next(
                    iterator
                )

            except StopIteration:
                break

            images = images.to(
                device
            )

            synchronize(
                device
            )

            start = time.perf_counter()

            _ = model(
                images
            )

            synchronize(
                device
            )

            end = time.perf_counter()

            latency = (
                end - start
            )

            batch_latencies.append(
                latency
            )

            total_images += (
                images.size(0)
            )

    total_time = sum(
        batch_latencies
    )

    average_batch_latency = mean(
        batch_latencies
    )

    average_image_latency = (
        total_time
        / total_images
    )

    throughput = (
        total_images
        / total_time
    )

    return (
        average_batch_latency,
        average_image_latency,
        throughput,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print(
        "\n=== Baseline Model Benchmark ===\n"
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    print(
        f"Loading model from: {MODEL_URI}"
    )

    model = mlflow.pytorch.load_model(
        MODEL_URI
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        "Model loaded successfully.\n"
    )

    test_loader = get_test_loader()

    total_parameters, trainable_parameters = (
        count_parameters(
            model
        )
    )

    model_size_mb = calculate_model_size(
        model
    )

    print(
        "Evaluating accuracy..."
    )

    accuracy = evaluate_accuracy(
        model,
        test_loader,
        device,
    )

    print(
        "Benchmarking inference..."
    )

    (
        average_batch_latency,
        average_image_latency,
        throughput,
    ) = benchmark_inference(
        model,
        test_loader,
        device,
    )

    print(
        "\n=== Benchmark Results ==="
    )

    print(
        f"Model URI: {MODEL_URI}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Precision: FP32"
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Model size: "
        f"{model_size_mb:.3f} MB"
    )

    print(
        f"Test accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Average batch latency: "
        f"{average_batch_latency * 1000:.3f} ms"
    )

    print(
        f"Average image latency: "
        f"{average_image_latency * 1000:.3f} ms"
    )

    print(
        f"Throughput: "
        f"{throughput:.2f} images/second"
    )

    print(
        "\nBaseline benchmark complete."
    )


if __name__ == "__main__":
    main()