import os
import platform
import time
from statistics import mean

import mlflow.pytorch
import numpy as np
import psutil
import torch
from sklearn.metrics import confusion_matrix
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


REGISTERED_MODEL_NAME = "industrial-baseline-cnn"
MODEL_ALIAS = "champion"

MODEL_URI = (
    f"models:/{REGISTERED_MODEL_NAME}"
    f"@{MODEL_ALIAS}"
)

BATCH_SIZES = [1, 8, 32, 64]

WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

MODEL_TEMP_PATH = (
    "artifacts/_industry_benchmark_model.pth"
)


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
# Synchronization
# ---------------------------------------------------------

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


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
# Percentiles
# ---------------------------------------------------------

def percentile(values, q):
    return float(
        np.percentile(
            values,
            q,
        )
    )


# ---------------------------------------------------------
# Parameter count
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Sparsity
# ---------------------------------------------------------

def calculate_sparsity(model):
    total = 0
    zero = 0

    for parameter in model.parameters():

        if parameter.ndim > 1:

            total += parameter.numel()

            zero += (
                parameter == 0
            ).sum().item()

    if total == 0:
        return 0.0

    return zero / total


# ---------------------------------------------------------
# Model size
# ---------------------------------------------------------

def get_model_size(model):
    os.makedirs(
        "artifacts",
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        MODEL_TEMP_PATH,
    )

    size_mb = (
        os.path.getsize(
            MODEL_TEMP_PATH
        )
        / 1024
        / 1024
    )

    os.remove(
        MODEL_TEMP_PATH
    )

    return size_mb


# ---------------------------------------------------------
# Process RAM
# ---------------------------------------------------------

def get_process_memory_mb():
    process = psutil.Process(
        os.getpid()
    )

    return (
        process.memory_info().rss
        / 1024
        / 1024
    )


# ---------------------------------------------------------
# Device memory
# ---------------------------------------------------------

def get_device_memory_mb(device):
    if device.type == "cuda":

        return (
            torch.cuda.memory_allocated()
            / 1024
            / 1024
        )

    if device.type == "mps":

        try:
            return (
                torch.mps.current_allocated_memory()
                / 1024
                / 1024
            )

        except Exception:
            return None

    return None


# ---------------------------------------------------------
# FLOPs / MACs
# ---------------------------------------------------------

def calculate_flops_macs(
    model,
    device,
    dtype,
):
    example = torch.randn(
        1,
        1,
        28,
        28,
        device=device,
        dtype=dtype,
    )

    try:
        flop_counter = FlopCountAnalysis(
            model,
            example,
        )

        flops = flop_counter.total()

        # Approximation:
        # 1 MAC is commonly treated as ~2 FLOPs.
        macs = flops / 2

        params = sum(
            p.numel()
            for p in model.parameters()
        )

        return (
            macs,
            flops,
            params,
        )

    except Exception as exc:
        print(
            f"FLOPs/MACs calculation skipped: {exc}"
        )

        return (
            None,
            None,
            None,
        )

# ---------------------------------------------------------
# Accuracy + class metrics
# ---------------------------------------------------------

def evaluate_model(
    model,
    dataset,
    device,
    dtype,
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

            images = images.to(
                device=device,
                dtype=dtype,
            )

            labels_device = labels.to(
                device
            )

            outputs = model(
                images
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions
                == labels_device
            ).sum().item()

            total += labels.size(0)

            all_labels.extend(
                labels.tolist()
            )

            all_predictions.extend(
                predictions
                .cpu()
                .tolist()
            )

    accuracy = correct / total

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    class_accuracy = []

    for class_id in range(10):

        denominator = matrix[
            class_id
        ].sum()

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
# Batch benchmark
# ---------------------------------------------------------

def benchmark_batch_size(
    model,
    dataset,
    device,
    dtype,
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

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------

    with torch.no_grad():

        for _ in range(
            WARMUP_ITERATIONS
        ):

            try:
                images, _ = next(
                    iterator
                )

            except StopIteration:

                iterator = iter(
                    loader
                )

                images, _ = next(
                    iterator
                )

            images = images.to(
                device=device,
                dtype=dtype,
            )

            _ = model(
                images
            )

            synchronize(
                device
            )

    # -----------------------------------------------------
    # Timed inference
    # -----------------------------------------------------

    iterator = iter(
        loader
    )

    latency_values = []

    total_images = 0

    with torch.no_grad():

        for _ in range(
            BENCHMARK_ITERATIONS
        ):

            try:
                images, _ = next(
                    iterator
                )

            except StopIteration:
                break

            images = images.to(
                device=device,
                dtype=dtype,
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
            mean(
                latency_values
            ),

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
# Cold start
# ---------------------------------------------------------

def measure_cold_inference(
    model,
    device,
    dtype,
):
    sample = torch.randn(
        1,
        1,
        28,
        28,
        device=device,
        dtype=dtype,
    )

    synchronize(
        device
    )

    start = time.perf_counter()

    with torch.no_grad():
        _ = model(sample)

    synchronize(
        device
    )

    end = time.perf_counter()

    return (
        end - start
    ) * 1000


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print(
        "\n===================================="
    )

    print(
        " INDUSTRY MODEL BENCHMARK"
    )

    print(
        "====================================\n"
    )

    device = get_device()

    dtype = torch.float32

    print(
        f"Python platform: "
        f"{platform.platform()}"
    )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"Device: "
        f"{device}"
    )

    print(
        f"Precision: FP32"
    )

    print(
        f"\nLoading model: "
        f"{MODEL_URI}"
    )

    ram_before = (
        get_process_memory_mb()
    )

    start_load = (
        time.perf_counter()
    )

    model = (
        mlflow.pytorch
        .load_model(
            MODEL_URI
        )
    )

    load_time = (
        time.perf_counter()
        - start_load
    )

    model = model.to(
        device
    )

    model.eval()

    ram_after = (
        get_process_memory_mb()
    )

    print(
        "Model loaded successfully."
    )

    dataset = get_dataset()

    total_parameters, trainable_parameters = (
        count_parameters(
            model
        )
    )

    model_size = (
        get_model_size(
            model
        )
    )

    sparsity = (
        calculate_sparsity(
            model
        )
    )

    cold_latency = (
        measure_cold_inference(
            model,
            device,
            dtype,
        )
    )

    print(
        "\nEvaluating model accuracy..."
    )

    (
        accuracy,
        matrix,
        class_accuracy,
    ) = evaluate_model(
        model,
        dataset,
        device,
        dtype,
    )

    print(
        "Running batch benchmarks..."
    )

    batch_results = []

    for batch_size in BATCH_SIZES:

        print(
            f"  Batch size: "
            f"{batch_size}"
        )

        result = (
            benchmark_batch_size(
                model,
                dataset,
                device,
                dtype,
                batch_size,
            )
        )

        batch_results.append(
            result
        )

    macs, flops, profile_params = (
        calculate_flops_macs(
            model,
            device,
            dtype,
        )
    )

    process_ram = (
        get_process_memory_mb()
    )

    device_memory = (
        get_device_memory_mb(
            device
        )
    )

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
        f"Device: "
        f"{device}"
    )

    print(
        f"Precision: "
        f"FP32"
    )

    print(
        f"Parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Model size: "
        f"{model_size:.3f} MB"
    )

    print(
        f"Sparsity: "
        f"{sparsity * 100:.2f}%"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Model load time: "
        f"{load_time * 1000:.2f} ms"
    )

    print(
        f"Cold inference latency: "
        f"{cold_latency:.3f} ms"
    )

    print(
        f"Process RAM before model: "
        f"{ram_before:.2f} MB"
    )

    print(
        f"Process RAM after model: "
        f"{ram_after:.2f} MB"
    )

    print(
        f"Current process RAM: "
        f"{process_ram:.2f} MB"
    )

    if device_memory is not None:

        print(
            f"Device allocated memory: "
            f"{device_memory:.2f} MB"
        )

    if macs is not None:

        print(
            f"MACs: "
            f"{macs:,.0f}"
        )

    if flops is not None:

        print(
            f"Approx FLOPs: "
            f"{flops:,.0f}"
        )

    print(
        "\n===================================="
    )

    print(
        " LATENCY / THROUGHPUT"
    )

    print(
        "===================================="
    )

    for result in batch_results:

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

    for index, value in enumerate(
        class_accuracy
    ):

        print(
            f"{index} "
            f"{class_names[index]:12s}: "
            f"{value * 100:.2f}%"
        )

    print(
        "\n===================================="
    )

    print(
        "Benchmark complete."
    )


if __name__ == "__main__":
    main()