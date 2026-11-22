import os
import platform
import time
from statistics import mean

import mlflow.pytorch
import numpy as np
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
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

BATCH_SIZES = [1, 8, 32, 64]

WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

FP32_TEMP_PATH = "artifacts/_industry_int8_fp32_temp.pth"
INT8_TEMP_PATH = "artifacts/_industry_int8_model.pth"


# ---------------------------------------------------------
# Quantization backend
# ---------------------------------------------------------

if "qnnpack" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "qnnpack"


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

def get_dataset():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,),
                (0.5,),
            ),
        ]
    )

    return datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def percentile(values, q):
    return float(np.percentile(values, q))


def get_process_memory_mb():
    process = psutil.Process(os.getpid())

    return (
        process.memory_info().rss
        / 1024
        / 1024
    )


def count_parameters(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def get_model_size_mb(
    model,
    path,
):
    os.makedirs(
        "artifacts",
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        path,
    )

    size_mb = (
        os.path.getsize(path)
        / 1024
        / 1024
    )

    return size_mb


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

def evaluate_model(
    model,
    dataset,
):
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    model.eval()

    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_labels.extend(
                labels.tolist()
            )

            all_predictions.extend(
                predictions.tolist()
            )

    accuracy = correct / total

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    class_accuracy = []

    for class_id in range(10):

        denominator = (
            matrix[class_id].sum()
        )

        if denominator == 0:
            value = 0.0

        else:
            value = (
                matrix[
                    class_id,
                    class_id,
                ]
                / denominator
            )

        class_accuracy.append(
            value
        )

    return (
        accuracy,
        matrix,
        class_accuracy,
    )


# ---------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------

def benchmark_batch_size(
    model,
    dataset,
    batch_size,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    iterator = iter(loader)

    model.eval()

    # Warm-up
    with torch.no_grad():

        for _ in range(
            WARMUP_ITERATIONS
        ):

            try:
                images, _ = next(iterator)

            except StopIteration:

                iterator = iter(loader)

                images, _ = next(iterator)

            _ = model(images)

    # Timed benchmark
    iterator = iter(loader)

    latency_values = []
    total_images = 0

    with torch.no_grad():

        for _ in range(
            BENCHMARK_ITERATIONS
        ):

            try:
                images, _ = next(iterator)

            except StopIteration:
                break

            start = time.perf_counter()

            _ = model(images)

            end = time.perf_counter()

            latency_ms = (
                end - start
            ) * 1000

            latency_values.append(
                latency_ms
            )

            total_images += (
                images.size(0)
            )

    total_time_seconds = (
        sum(latency_values)
        / 1000
    )

    throughput = (
        total_images
        / total_time_seconds
    )

    return {
        "batch_size":
            batch_size,

        "mean_latency_ms":
            mean(latency_values),

        "p50_latency_ms":
            percentile(
                latency_values,
                50,
            ),

        "p95_latency_ms":
            percentile(
                latency_values,
                95,
            ),

        "p99_latency_ms":
            percentile(
                latency_values,
                99,
            ),

        "throughput_img_s":
            throughput,
    }


# ---------------------------------------------------------
# Cold inference
# ---------------------------------------------------------

def measure_cold_inference(
    model,
):
    sample = torch.randn(
        1,
        1,
        28,
        28,
    )

    start = (
        time.perf_counter()
    )

    with torch.no_grad():

        _ = model(
            sample
        )

    end = (
        time.perf_counter()
    )

    return (
        end - start
    ) * 1000


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
        "\n===================================="
    )

    print(
        " INDUSTRY INT8 MODEL BENCHMARK"
    )

    print(
        "====================================\n"
    )

    print(
        f"Python platform: "
        f"{platform.platform()}"
    )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"Device: CPU"
    )

    print(
        f"Quantization engine: "
        f"{torch.backends.quantized.engine}"
    )

    print(
        f"Quantization type: "
        f"Dynamic INT8"
    )

    print(
        f"\nLoading model: "
        f"{MODEL_URI}"
    )

    ram_before = (
        get_process_memory_mb()
    )

    load_start = (
        time.perf_counter()
    )

    fp32_model = (
        mlflow.pytorch.load_model(
            MODEL_URI
        )
    )

    load_time = (
        time.perf_counter()
        - load_start
    )

    fp32_model = (
        fp32_model.cpu()
    )

    fp32_model.eval()

    print(
        "FP32 model loaded successfully."
    )

    dataset = get_dataset()

    fp32_parameters, fp32_trainable = (
        count_parameters(
            fp32_model
        )
    )

    fp32_size = (
        get_model_size_mb(
            fp32_model,
            FP32_TEMP_PATH,
        )
    )

    print(
        "\nEvaluating FP32 CPU baseline..."
    )

    (
        fp32_accuracy,
        fp32_matrix,
        fp32_class_accuracy,
    ) = evaluate_model(
        fp32_model,
        dataset,
    )

    fp32_cold_latency = (
        measure_cold_inference(
            fp32_model
        )
    )

    fp32_batch_results = []

    for batch_size in BATCH_SIZES:

        print(
            f"  FP32 batch size: "
            f"{batch_size}"
        )

        result = (
            benchmark_batch_size(
                fp32_model,
                dataset,
                batch_size,
            )
        )

        fp32_batch_results.append(
            result
        )

    # -----------------------------------------------------
    # Quantize
    # -----------------------------------------------------

    int8_model = (
        quantize_model(
            fp32_model
        )
    )

    int8_model.eval()

    ram_after_quantization = (
        get_process_memory_mb()
    )

    int8_parameters, int8_trainable = (
        count_parameters(
            int8_model
        )
    )

    int8_size = (
        get_model_size_mb(
            int8_model,
            INT8_TEMP_PATH,
        )
    )

    print(
        "\nEvaluating INT8 model..."
    )

    (
        int8_accuracy,
        int8_matrix,
        int8_class_accuracy,
    ) = evaluate_model(
        int8_model,
        dataset,
    )

    int8_cold_latency = (
        measure_cold_inference(
            int8_model
        )
    )

    int8_batch_results = []

    for batch_size in BATCH_SIZES:

        print(
            f"  INT8 batch size: "
            f"{batch_size}"
        )

        result = (
            benchmark_batch_size(
                int8_model,
                dataset,
                batch_size,
            )
        )

        int8_batch_results.append(
            result
        )

    current_ram = (
        get_process_memory_mb()
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print(
        "\n===================================="
    )

    print(
        " MODEL INFORMATION"
    )

    print(
        "===================================="
    )

    print(
        f"Model URI: "
        f"{MODEL_URI}"
    )

    print(
        f"Device: CPU"
    )

    print(
        f"Quantization engine: "
        f"{torch.backends.quantized.engine}"
    )

    print(
        f"Parameters before quantization: "
        f"{fp32_parameters:,}"
    )

    print(
        f"Trainable parameters before: "
        f"{fp32_trainable:,}"
    )

    print(
        f"FP32 model size: "
        f"{fp32_size:.3f} MB"
    )

    print(
        f"INT8 model size: "
        f"{int8_size:.3f} MB"
    )

    size_reduction = (
        (
            fp32_size
            - int8_size
        )
        / fp32_size
        * 100
    )

    print(
        f"Model size reduction: "
        f"{size_reduction:.2f}%"
    )

    print(
        f"FP32 accuracy: "
        f"{fp32_accuracy * 100:.2f}%"
    )

    print(
        f"INT8 accuracy: "
        f"{int8_accuracy * 100:.2f}%"
    )

    print(
        f"Accuracy change: "
        f"{(fp32_accuracy - int8_accuracy) * 100:.2f} "
        f"percentage points"
    )

    print(
        f"Model load time: "
        f"{load_time * 1000:.2f} ms"
    )

    print(
        f"FP32 cold inference: "
        f"{fp32_cold_latency:.3f} ms"
    )

    print(
        f"INT8 cold inference: "
        f"{int8_cold_latency:.3f} ms"
    )

    print(
        f"Process RAM before model: "
        f"{ram_before:.2f} MB"
    )

    print(
        f"Process RAM after quantization: "
        f"{ram_after_quantization:.2f} MB"
    )

    print(
        f"Current process RAM: "
        f"{current_ram:.2f} MB"
    )

    # -----------------------------------------------------
    # FP32 latency
    # -----------------------------------------------------

    print(
        "\n===================================="
    )

    print(
        " FP32 CPU LATENCY / THROUGHPUT"
    )

    print(
        "===================================="
    )

    for result in (
        fp32_batch_results
    ):

        print(
            f"\nBatch size "
            f"{result['batch_size']}"
        )

        print(
            f"Mean latency: "
            f"{result['mean_latency_ms']:.3f} ms"
        )

        print(
            f"P50 latency: "
            f"{result['p50_latency_ms']:.3f} ms"
        )

        print(
            f"P95 latency: "
            f"{result['p95_latency_ms']:.3f} ms"
        )

        print(
            f"P99 latency: "
            f"{result['p99_latency_ms']:.3f} ms"
        )

        print(
            f"Throughput: "
            f"{result['throughput_img_s']:.2f} "
            f"images/sec"
        )

    # -----------------------------------------------------
    # INT8 latency
    # -----------------------------------------------------

    print(
        "\n===================================="
    )

    print(
        " INT8 LATENCY / THROUGHPUT"
    )

    print(
        "===================================="
    )

    for result in (
        int8_batch_results
    ):

        print(
            f"\nBatch size "
            f"{result['batch_size']}"
        )

        print(
            f"Mean latency: "
            f"{result['mean_latency_ms']:.3f} ms"
        )

        print(
            f"P50 latency: "
            f"{result['p50_latency_ms']:.3f} ms"
        )

        print(
            f"P95 latency: "
            f"{result['p95_latency_ms']:.3f} ms"
        )

        print(
            f"P99 latency: "
            f"{result['p99_latency_ms']:.3f} ms"
        )

        print(
            f"Throughput: "
            f"{result['throughput_img_s']:.2f} "
            f"images/sec"
        )

    # -----------------------------------------------------
    # Per-class accuracy
    # -----------------------------------------------------

    print(
        "\n===================================="
    )

    print(
        " PER-CLASS ACCURACY"
    )

    print(
        "===================================="
    )

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"{index} "
            f"{class_name:12s}: "
            f"FP32 "
            f"{fp32_class_accuracy[index] * 100:.2f}% "
            f"| INT8 "
            f"{int8_class_accuracy[index] * 100:.2f}%"
        )

    # -----------------------------------------------------
    # Direct comparison
    # -----------------------------------------------------

    print(
        "\n===================================="
    )

    print(
        " INT8 VS FP32 CPU"
    )

    print(
        "===================================="
    )

    for (
        fp32_result,
        int8_result,
    ) in zip(
        fp32_batch_results,
        int8_batch_results,
    ):

        speedup = (
            fp32_result[
                "mean_latency_ms"
            ]
            / int8_result[
                "mean_latency_ms"
            ]
        )

        print(
            f"Batch "
            f"{fp32_result['batch_size']}: "
            f"{speedup:.2f}x "
            f"latency speedup"
        )

    print(
        "\nINT8 benchmark complete."
    )


if __name__ == "__main__":
    main()