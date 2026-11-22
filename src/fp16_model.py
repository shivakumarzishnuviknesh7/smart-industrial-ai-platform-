import copy
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

FP16_OUTPUT_PATH = "artifacts/fp16_model.pth"


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
    dtype,
):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device=device,
                dtype=dtype,
            )

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
# Benchmark
# ---------------------------------------------------------

def benchmark_inference(
    model,
    test_loader,
    device,
    dtype,
):
    model.eval()

    iterator = iter(test_loader)

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------

    with torch.no_grad():

        for _ in range(
            NUM_WARMUP_BATCHES
        ):

            try:
                images, _ = next(iterator)

            except StopIteration:
                iterator = iter(test_loader)
                images, _ = next(iterator)

            images = images.to(
                device=device,
                dtype=dtype,
            )

            _ = model(images)

            synchronize(device)

    # -----------------------------------------------------
    # Benchmark
    # -----------------------------------------------------

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

            images = images.to(
                device=device,
                dtype=dtype,
            )

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
# Main
# ---------------------------------------------------------

def main():
    print(
        "\n=== FP16 Optimization Experiment ===\n"
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    if device.type == "cpu":
        print(
            "\nWARNING:"
            "\nFP16 CPU execution is not the target "
            "for this experiment."
        )

    print(
        f"\nLoading champion model: "
        f"{MODEL_URI}"
    )

    baseline_model = mlflow.pytorch.load_model(
        MODEL_URI
    )

    baseline_model = baseline_model.to(
        device
    )

    baseline_model.eval()

    print(
        "Model loaded successfully."
    )

    test_loader = get_test_loader()

    # -----------------------------------------------------
    # FP32 benchmark
    # -----------------------------------------------------

    print(
        "\nEvaluating FP32 model..."
    )

    baseline_accuracy = evaluate_accuracy(
        baseline_model,
        test_loader,
        device,
        torch.float32,
    )

    (
        baseline_batch_latency,
        baseline_image_latency,
        baseline_throughput,
    ) = benchmark_inference(
        baseline_model,
        test_loader,
        device,
        torch.float32,
    )

    baseline_temp_path = (
        "artifacts/_fp32_temp.pth"
    )

    baseline_size = get_model_size_mb(
        baseline_model,
        baseline_temp_path,
    )

    os.remove(
        baseline_temp_path
    )

    # -----------------------------------------------------
    # FP16 conversion
    # -----------------------------------------------------

    print(
        "\nConverting model to FP16..."
    )

    fp16_model = copy.deepcopy(
        baseline_model
    )

    fp16_model = fp16_model.half()

    fp16_model.eval()

    # -----------------------------------------------------
    # FP16 evaluation
    # -----------------------------------------------------

    print(
        "Evaluating FP16 model..."
    )

    fp16_accuracy = evaluate_accuracy(
        fp16_model,
        test_loader,
        device,
        torch.float16,
    )

    (
        fp16_batch_latency,
        fp16_image_latency,
        fp16_throughput,
    ) = benchmark_inference(
        fp16_model,
        test_loader,
        device,
        torch.float16,
    )

    fp16_size = get_model_size_mb(
        fp16_model,
        FP16_OUTPUT_PATH,
    )

    # -----------------------------------------------------
    # Differences
    # -----------------------------------------------------

    accuracy_drop = (
        baseline_accuracy
        - fp16_accuracy
    )

    size_reduction_mb = (
        baseline_size
        - fp16_size
    )

    size_reduction_percent = (
        size_reduction_mb
        / baseline_size
        * 100
    )

    speedup = (
        baseline_batch_latency
        / fp16_batch_latency
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print(
        "\n================================"
    )

    print(
        "          FP16 RESULTS"
    )

    print(
        "================================"
    )

    print(
        "\n--- FP32 Baseline ---"
    )

    print(
        f"Accuracy: "
        f"{baseline_accuracy * 100:.2f}%"
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
        f"{baseline_throughput:.2f} images/sec"
    )

    print(
        "\n--- FP16 Model ---"
    )

    print(
        f"Accuracy: "
        f"{fp16_accuracy * 100:.2f}%"
    )

    print(
        f"Model size: "
        f"{fp16_size:.3f} MB"
    )

    print(
        f"Batch latency: "
        f"{fp16_batch_latency * 1000:.3f} ms"
    )

    print(
        f"Image latency: "
        f"{fp16_image_latency * 1000:.3f} ms"
    )

    print(
        f"Throughput: "
        f"{fp16_throughput:.2f} images/sec"
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
        f"Model size reduction: "
        f"{size_reduction_mb:.3f} MB"
    )

    print(
        f"Model size reduction: "
        f"{size_reduction_percent:.2f}%"
    )

    print(
        f"Latency speedup: "
        f"{speedup:.2f}x"
    )

    print(
        f"\nSaved FP16 model to: "
        f"{FP16_OUTPUT_PATH}"
    )

    print(
        "\nFP16 experiment complete."
    )


if __name__ == "__main__":
    main()