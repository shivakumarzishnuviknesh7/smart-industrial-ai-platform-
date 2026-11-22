import os
import time
from statistics import mean

import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
if "qnnpack" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "qnnpack"

print(
    "Quantization engine:",
    torch.backends.quantized.engine,
)

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

OUTPUT_PATH = "artifacts/quantized_model.pth"


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
):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

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

            _ = model(images)

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

            start = time.perf_counter()

            _ = model(images)

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
# Dynamic INT8 quantization
# ---------------------------------------------------------

def quantize_model(model):
    print(
        "\nApplying dynamic INT8 quantization "
        "to Linear layers..."
    )

    quantized_model = (
        torch.ao.quantization.quantize_dynamic(
            model,
            {
                nn.Linear,
            },
            dtype=torch.qint8,
        )
    )

    return quantized_model


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print(
        "\n=== INT8 Quantization Experiment ===\n"
    )

    # IMPORTANT:
    # Dynamic quantized PyTorch operators run on CPU.
    device = torch.device("cpu")

    print(
        f"Benchmark device: {device}"
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

    # -----------------------------------------------------
    # FP32 baseline on CPU
    # -----------------------------------------------------

    print(
        "\nEvaluating FP32 CPU baseline..."
    )

    baseline_accuracy = evaluate_accuracy(
        model,
        test_loader,
    )

    (
        baseline_batch_latency,
        baseline_image_latency,
        baseline_throughput,
    ) = benchmark_inference(
        model,
        test_loader,
    )

    baseline_temp_path = (
        "artifacts/"
        "_quantization_fp32_temp.pth"
    )

    baseline_size = get_model_size_mb(
        model,
        baseline_temp_path,
    )

    os.remove(
        baseline_temp_path
    )

    # -----------------------------------------------------
    # Quantization
    # -----------------------------------------------------

    quantized_model = quantize_model(
        model
    )

    quantized_model.eval()

    # -----------------------------------------------------
    # Quantized evaluation
    # -----------------------------------------------------

    print(
        "\nEvaluating INT8 quantized model..."
    )

    quantized_accuracy = evaluate_accuracy(
        quantized_model,
        test_loader,
    )

    (
        quantized_batch_latency,
        quantized_image_latency,
        quantized_throughput,
    ) = benchmark_inference(
        quantized_model,
        test_loader,
    )

    quantized_size = get_model_size_mb(
        quantized_model,
        OUTPUT_PATH,
    )

    # -----------------------------------------------------
    # Differences
    # -----------------------------------------------------

    accuracy_drop = (
        baseline_accuracy
        - quantized_accuracy
    )

    size_reduction_mb = (
        baseline_size
        - quantized_size
    )

    size_reduction_percent = (
        size_reduction_mb
        / baseline_size
        * 100
    )

    speedup = (
        baseline_batch_latency
        / quantized_batch_latency
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print(
        "\n================================"
    )

    print(
        "      QUANTIZATION RESULTS"
    )

    print(
        "================================"
    )

    print(
        "\n--- FP32 CPU Baseline ---"
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
        "\n--- INT8 Quantized Model ---"
    )

    print(
        f"Accuracy: "
        f"{quantized_accuracy * 100:.2f}%"
    )

    print(
        f"Model size: "
        f"{quantized_size:.3f} MB"
    )

    print(
        f"Batch latency: "
        f"{quantized_batch_latency * 1000:.3f} ms"
    )

    print(
        f"Image latency: "
        f"{quantized_image_latency * 1000:.3f} ms"
    )

    print(
        f"Throughput: "
        f"{quantized_throughput:.2f} images/sec"
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
        f"\nSaved quantized model to: "
        f"{OUTPUT_PATH}"
    )

    print(
        "\nQuantization experiment complete."
    )


if __name__ == "__main__":
    main()