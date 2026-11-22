# Smart Industrial AI Platform

An end-to-end MLOps and edge-AI platform built with PyTorch, MLflow, FastAPI, Docker, Kubernetes, Flux GitOps, Prometheus, and Grafana.

The project demonstrates the complete machine-learning lifecycle:

**training → experiment tracking → model registry → promotion → inference serving → containerization → Kubernetes deployment → GitOps → monitoring → rollback → model optimization → benchmarking**

---

## 1. Project Overview

The Smart Industrial AI Platform is designed as a practical MLOps system for managing and deploying machine-learning models in a reproducible and production-oriented way.

The project focuses on the engineering challenges involved in taking a trained ML model from experimentation to deployment.

The platform includes:

- PyTorch model training
- FashionMNIST image classification
- MLflow experiment tracking
- MLflow Model Registry
- Model aliases and promotion
- FastAPI inference service
- Docker containerization
- Kubernetes deployment
- Flux GitOps
- GitLab CI/CD
- Prometheus monitoring
- Grafana dashboards
- Deployment rollback
- CLI-based model operations
- FP32 benchmarking
- FP16 optimization
- INT8 quantization
- Unstructured pruning
- Edge-inference benchmarking

---

# 2. Architecture

```text
                       ┌─────────────────────┐
                       │   FashionMNIST Data │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │   PyTorch Training  │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │       MLflow        │
                       │ Experiments + Runs  │
                       └──────────┬──────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ MLflow Model Registry   │
                     │ industrial-baseline-cnn │
                     └────────────┬────────────┘
                                  │
                           champion alias
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │      FastAPI        │
                       │  Inference Service  │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │       Docker        │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │     Kubernetes      │
                       └──────────┬──────────┘
                                  │
                      Flux GitOps │
                                  ▼
                       ┌─────────────────────┐
                       │ Deployment + Service│
                       └──────────┬──────────┘
                                  │
                        Metrics   │
                                  ▼
                 ┌──────────────────────────────┐
                 │ Prometheus + Grafana         │
                 │ Monitoring and Observability │
                 └──────────────────────────────┘
```

---

# 3. Technology Stack

## Machine Learning

- Python
- PyTorch
- torchvision
- FashionMNIST
- NumPy
- scikit-learn

## Experiment Tracking

- MLflow
- MLflow Model Registry
- Model aliases
- SQLite backend
- Local artifact storage

## API

- FastAPI
- Uvicorn
- Pydantic

## Containerization

- Docker

## Orchestration

- Kubernetes
- Docker Desktop Kubernetes

## GitOps

- Flux CD

## CI/CD

- GitLab CI

## Monitoring

- Prometheus
- Grafana
- kube-prometheus-stack

## Model Optimization

- FP16
- Dynamic INT8 quantization
- QNNPACK
- Unstructured pruning
- PyTorch MPS
- CPU benchmarking

---

# 4. Dataset

The platform currently uses FashionMNIST.

FashionMNIST contains grayscale images with 10 clothing categories.

Image dimensions:

```text
1 × 28 × 28
```

Classes:

```text
0  T-shirt/top
1  Trouser
2  Pullover
3  Dress
4  Coat
5  Sandal
6  Shirt
7  Sneaker
8  Bag
9  Ankle boot
```

Dataset split used in this project:

```text
Training:   48,000
Validation: 12,000
Test:       10,000
```

---

# 5. Model

The project uses a custom convolutional neural network implemented in PyTorch.

The model contains:

- convolution layers
- activation functions
- pooling
- fully connected layers
- 10-class output layer

Total parameter count:

```text
421,642 parameters
```

Baseline model size:

```text
1.612 MB
```

---

# 6. Training

The model is trained using PyTorch.

Training is configured using:

```text
configs/baseline.yaml
```

Example:

```bash
python src/train.py --config configs/baseline.yaml
```

The training pipeline performs:

```text
Dataset loading
      ↓
Training
      ↓
Validation
      ↓
Best-model selection
      ↓
Test evaluation
      ↓
MLflow logging
      ↓
Model registration
```

---

# 7. Baseline Model Performance

The best baseline model achieved:

```text
Test accuracy: 90.58%
```

Training results:

```text
Epoch 1 validation accuracy: 86.81%
Epoch 2 validation accuracy: 88.90%
Epoch 3 validation accuracy: 90.32%
Epoch 4 validation accuracy: 91.22%
Epoch 5 validation accuracy: 91.20%
```

Best model:

```text
Epoch 4
```

---

# 8. MLflow

MLflow is used for:

- experiment tracking
- parameters
- metrics
- artifacts
- model storage
- model registration
- model versioning
- model aliases

The registered model is:

```text
industrial-baseline-cnn
```

The production-style alias used by the inference service is:

```text
champion
```

The application therefore loads:

```text
models:/industrial-baseline-cnn@champion
```

This separates model deployment from hardcoded model-version numbers.

---

# 9. Running MLflow

The local MLflow server runs on port:

```text
5001
```

Example:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5001 \
  --allowed-hosts "localhost:5001,127.0.0.1:5001" \
  --serve-artifacts \
  --artifacts-destination ./mlartifacts
```

Set the tracking URI:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```

MLflow UI:

```text
http://127.0.0.1:5001
```

---

# 10. Model Promotion

Model promotion is implemented through MLflow Model Registry aliases.

For example:

```text
Model version 2
      ↓
alias = champion
```

The serving application does not need to know the specific numeric model version.

It always loads:

```text
industrial-baseline-cnn@champion
```

This supports controlled promotion and rollback.

---

# 11. FastAPI Model Serving

The model is served through FastAPI.

Main serving file:

```text
src/serve.py
```

Start locally using:

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

# 12. API Endpoints

## Root

```text
GET /
```

## Health

```text
GET /health
```

Example:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model": "industrial-baseline-cnn",
  "alias": "champion"
}
```

## Model Information

```text
GET /model
```

## Prediction

```text
POST /predict
```

Prediction output:

```json
{
  "predicted_class": 3,
  "predicted_label": "Dress",
  "confidence": 0.94,
  "model_name": "industrial-baseline-cnn",
  "model_alias": "champion"
}
```

## Prometheus Metrics

```text
GET /metrics/
```

---

# 13. Prometheus Metrics

The FastAPI service exposes metrics including:

```text
prediction_requests_total
prediction_errors_total
prediction_latency_seconds
```

These metrics are scraped by Prometheus.

---

# 14. Docker

The FastAPI inference application is containerized using Docker.

Build:

```bash
docker build -t smart-industrial-ai-platform .
```

Run:

```bash
docker run -p 8000:8000 smart-industrial-ai-platform
```

Docker packages:

```text
Python runtime
Application code
Dependencies
Inference service
Model-loading logic
```

The model itself is not conceptually "converted into Docker".

Docker provides the runtime environment in which inference runs.

---

# 15. GitLab Container Registry

Container images have been built through CI and pushed to:

```text
registry.gitlab.com/zishnu-mlops/smart-industrial-ai-platform
```

Deployment manifests reference versioned container tags.

Example:

```text
580ef996
```

This enables reproducible deployment and rollback.

---

# 16. Kubernetes

The platform runs locally on Docker Desktop Kubernetes.

Kubernetes resources include:

- Deployment
- Service
- ConfigMap
- imagePullSecrets
- startup probe
- readiness probe
- liveness probe
- ServiceMonitor

Application manifests are stored under:

```text
clusters/docker-desktop/apps/smart-industrial-ai-platform/
```

---

# 17. GitOps with Flux

Flux is used to manage Kubernetes using Git as the source of truth.

Deployment flow:

```text
Developer changes Kubernetes YAML
           ↓
Git commit
           ↓
Git push
           ↓
Flux detects change
           ↓
Flux reconciles cluster
           ↓
Kubernetes deployment updated
```

This removes the need for manually running:

```text
kubectl apply
```

for normal GitOps deployment operations.

Manual reconciliation can be triggered with:

```bash
flux reconcile kustomization flux-system --with-source
```

---

# 18. Kubernetes Health Probes

The application contains:

```text
startupProbe
readinessProbe
livenessProbe
```

These allow Kubernetes to determine whether the model-serving application:

- has successfully started
- is ready to receive traffic
- is still healthy

This is especially important because loading ML models can take longer than starting a normal web application.

---

# 19. Prometheus

Prometheus is installed using kube-prometheus-stack.

Port forward:

```bash
kubectl port-forward \
  -n monitoring \
  service/kube-prometheus-stack-prometheus \
  9090:9090
```

Open:

```text
http://localhost:9090
```

---

# 20. Grafana

Grafana is used to visualize inference metrics.

Port forward:

```bash
kubectl port-forward \
  -n monitoring \
  service/kube-prometheus-stack-grafana \
  3000:80
```

Open:

```text
http://localhost:3000
```

Metrics monitored include:

- prediction requests
- prediction errors
- inference latency

---

# 21. Rollback

The platform supports GitOps-based deployment rollback.

Example:

```text
Current image
580ef996
    ↓
rollback
bcee6052
```

The deployment YAML is modified and committed.

Flux then detects the change and automatically updates the Kubernetes Deployment.

This provides:

```text
Git history
+
versioned images
+
declarative rollback
```

---

# 22. Platform CLI

The project includes:

```text
platform_cli.py
```

Example operations include:

```bash
python platform_cli.py status
```

```bash
python platform_cli.py promote --version 2
```

```bash
python platform_cli.py rollback --image-tag bcee6052
```

The CLI provides a simple self-service interface for platform operations.

---

# 23. Model Optimization

The project contains multiple model-optimization experiments.

These are designed to investigate the trade-offs between:

```text
Accuracy
Model size
Latency
Throughput
Memory
Precision
Hardware
```

Optimization experiments include:

- FP32 baseline
- FP16
- INT8 dynamic quantization
- unstructured pruning

Structured pruning and ONNX/TensorRT deployment are planned future stages.

---

# 24. Industry Benchmarking

The project includes detailed benchmarking scripts.

```text
src/benchmark_industry.py
src/benchmark_industry_fp16.py
src/benchmark_industry_int8.py
```

Metrics include:

- accuracy
- parameter count
- model size
- sparsity
- model load time
- cold inference latency
- process RAM
- device memory
- MACs
- FLOPs
- mean latency
- P50 latency
- P95 latency
- P99 latency
- throughput
- batch-size scaling
- per-class accuracy

Batch sizes tested:

```text
1
8
32
64
```

---

# 25. FP32 Benchmark

Hardware:

```text
Apple Silicon
PyTorch MPS
FP32
```

Results:

```text
Accuracy:           90.58%
Parameters:         421,642
Model size:         1.612 MB
Device memory:      1.61 MB
MACs:               2,120,576
Approx FLOPs:       4,241,152
Model load time:    92.82 ms
Cold inference:     352.786 ms
```

Batch-size-1 performance:

```text
Mean:       0.419 ms
P50:        0.378 ms
P95:        0.690 ms
P99:        0.745 ms
Throughput: 2,385.86 images/sec
```

Batch-size-64:

```text
Mean latency: 1.240 ms
Throughput:   51,599.81 images/sec
```

---

# 26. FP16 Benchmark

The FP32 model was converted to FP16.

Results:

```text
Accuracy:           90.58%
Model size:         0.808 MB
Device memory:      0.80 MB
Parameters:         421,642
MACs:               2,120,576
Approx FLOPs:       4,241,152
```

FP16 therefore achieved approximately:

```text
50% model-size reduction
50% device-memory reduction
0% overall accuracy loss
```

Batch-size-1:

```text
Mean latency: 0.354 ms
P50:          0.344 ms
P95:          0.402 ms
P99:          0.532 ms
Throughput:   2,822.89 images/sec
```

Batch-size-64:

```text
Mean latency: 1.129 ms
Throughput:   56,708.14 images/sec
```

Compared with FP32, FP16 provided a strong balance between compression, accuracy, and inference performance.

---

# 27. INT8 Quantization

Dynamic INT8 quantization was performed on the model's Linear layers.

Backend:

```text
QNNPACK
```

Device:

```text
CPU
```

The CPU FP32 model was used as the fair baseline.

Results:

```text
FP32 size:     1.612 MB
INT8 size:     0.462 MB

Size reduction:
71.38%
```

Accuracy:

```text
FP32: 90.58%
INT8: 90.57%
```

Accuracy loss:

```text
0.01 percentage points
```

---

# 28. INT8 Performance

Batch size 1:

```text
FP32:
0.616 ms

INT8:
0.184 ms

Speedup:
3.36x
```

Throughput:

```text
FP32:
1,623 images/sec

INT8:
5,448 images/sec
```

Batch size 8:

```text
1.39x latency speedup
```

Batch size 32:

```text
1.25x latency speedup
```

Batch size 64:

```text
~1.00x
```

This demonstrates that quantization benefits depend strongly on:

- model architecture
- quantized layer types
- batch size
- hardware
- inference backend

Only the Linear layers are dynamically quantized in this experiment, while the convolution layers remain FP32.

---

# 29. Unstructured Pruning

A 30% L1 unstructured pruning experiment was performed on eligible Conv2d and Linear weights.

Results:

```text
Baseline accuracy: 90.58%
Pruned accuracy:   89.69%

Sparsity:
0% → 30%
```

However:

```text
Model size:
1.612 MB → 1.612 MB
```

and inference did not become faster.

This demonstrates an important engineering principle:

> Zeroing individual weights does not automatically produce faster inference on dense hardware.

Unstructured pruning requires sparse storage, sparse kernels, or hardware acceleration to provide real deployment benefits.

---

# 30. Optimization Comparison

| Variant | Device | Accuracy | Size | Key Result |
|---|---|---:|---:|---|
| FP32 | MPS | 90.58% | 1.612 MB | Baseline |
| FP16 | MPS | 90.58% | 0.808 MB | ~50% smaller |
| INT8 | CPU/QNNPACK | 90.57% | 0.462 MB | ~71% smaller |
| 30% Unstructured Pruning | MPS | 89.69% | 1.612 MB | No dense size reduction |

Latency values should only be compared directly when the same execution device and backend are used.

---

# 31. Per-Class Baseline Accuracy

```text
T-shirt/top : 89.30%
Trouser     : 97.00%
Pullover    : 86.90%
Dress       : 91.70%
Coat        : 83.20%
Sandal      : 96.70%
Shirt       : 70.70%
Sneaker     : 95.80%
Bag         : 97.60%
Ankle boot  : 96.90%
```

The Shirt class is currently the most difficult category for the model.

This provides an additional quality signal when evaluating future optimization methods.

---

# 32. Edge AI Deployment Strategy

The intended optimization and deployment flow is:

```text
PyTorch training
      ↓
model validation
      ↓
MLflow registration
      ↓
optimization
      ↓
FP16 / INT8 / pruning
      ↓
ONNX
      ↓
ONNX Runtime
      ↓
TensorRT
      ↓
Docker
      ↓
NVIDIA edge hardware
```

Docker is used to package the runtime environment.

ONNX provides a portable model representation.

TensorRT can then optimize supported models for NVIDIA GPUs using techniques such as:

- kernel optimization
- layer fusion
- FP16
- INT8
- memory optimization
- Tensor Core acceleration

---

# 33. Apple Silicon Development

Current optimization experiments were performed locally on Apple Silicon.

Available execution paths include:

```text
CPU
MPS
```

NVIDIA-specific technologies such as:

```text
CUDA
TensorRT
Tensor Cores
```

cannot be executed natively on Apple Silicon.

TensorRT benchmarking therefore belongs to a future NVIDIA target environment.

---

# 34. Repository Structure

```text
smart-industrial-ai-platform/
│
├── artifacts/
│
├── clusters/
│   └── docker-desktop/
│       ├── flux-system/
│       └── apps/
│           └── smart-industrial-ai-platform/
│
├── configs/
│   └── baseline.yaml
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── promote_model.py
│   ├── serve.py
│   ├── test_api.py
│   ├── prune_model.py
│   ├── quantize_model.py
│   ├── fp16_model.py
│   ├── benchmark_model.py
│   ├── benchmark_industry.py
│   ├── benchmark_industry_fp16.py
│   └── benchmark_industry_int8.py
│
├── tests/
│   └── test_model.py
│
├── platform_cli.py
├── Dockerfile
├── requirements.txt
├── pytest.ini
├── .gitlab-ci.yml
├── .gitignore
└── README.md
```

Generated data, local environments, MLflow databases, model artifacts, and secrets should remain excluded using `.gitignore`.

---

# 35. Installation

Clone the repository:

```bash
git clone https://github.com/shivakumarzishnuviknesh7/smart-industrial-ai-platform-.git
```

Enter the directory:

```bash
cd smart-industrial-ai-platform-
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 36. Running the Project

Start MLflow.

Then configure:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```

Train:

```bash
python src/train.py --config configs/baseline.yaml
```

Run the API:

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

---

# 37. Running Benchmarks

FP32:

```bash
python -u src/benchmark_industry.py
```

FP16:

```bash
python -u src/benchmark_industry_fp16.py
```

INT8:

```bash
python -u src/benchmark_industry_int8.py
```

---

# 38. Key Engineering Learnings

This project demonstrates several practical MLOps principles.

### Model performance alone is not enough

A deployment decision should evaluate:

```text
Accuracy
Latency
P95/P99 latency
Throughput
Memory
Model size
Hardware compatibility
Cold-start time
```

### Quantization is hardware dependent

INT8 does not automatically provide the same performance benefit on every CPU or GPU.

### FP16 can provide an excellent trade-off

In this project FP16 achieved:

```text
No overall accuracy loss
~50% lower model size
~50% lower device memory
Improved warm inference performance
```

### Sparse weights do not guarantee speed

Unstructured pruning produced 30% sparsity but did not reduce dense model size or inference latency.

### Batch size changes optimization behavior

INT8 showed strong improvement at batch size 1 but almost no advantage at batch size 64.

### GitOps improves reproducibility

The desired Kubernetes state is version-controlled and reconciled automatically by Flux.

### Model deployment and model promotion are separate

Container versions are managed through Git/Kubernetes.

ML model versions are managed through MLflow aliases.

This separation enables safer deployment and rollback strategies.

---

# 39. Future Work

Planned extensions include:

- structured channel pruning
- fine-tuning after structured pruning
- ONNX export
- ONNX Runtime benchmark
- ONNX model validation
- TensorRT engine generation
- NVIDIA GPU benchmarking
- CUDA inference
- Tensor Core FP16 execution
- INT8 calibration
- TensorRT INT8
- GPU memory profiling
- power-consumption monitoring
- API load testing
- model-drift monitoring
- automated optimization pipeline
- automated model-promotion gates
- performance regression tests
- CI-based benchmark comparison

---

# 40. Project Goal

The goal of this project is not simply to train a neural network.

The goal is to demonstrate how machine-learning models can be:

```text
trained
tracked
versioned
tested
optimized
served
containerized
deployed
monitored
promoted
rolled back
and benchmarked
```

using modern MLOps engineering practices.

The project is designed as a practical foundation for industrial AI, edge AI, machine-learning infrastructure, and MLOps engineering workflows.
