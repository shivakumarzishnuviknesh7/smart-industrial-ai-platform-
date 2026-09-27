import argparse
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import yaml
from mlflow.models import infer_signature

from data import get_dataloaders
from evaluate import evaluate_model
from model import create_model


def load_config(config_path: str):
    """
    Load YAML configuration.
    """
    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def get_device():
    """
    Select the best available compute device.

    Priority:
        CUDA
        Apple Silicon MPS
        CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def configure_mlflow():
    """
    Configure MLflow tracking and model registry.

    Local Mac default:
        http://127.0.0.1:5001

    Can be overridden using:

        MLFLOW_TRACKING_URI=<uri>
    """

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5001",
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_registry_uri(
        tracking_uri
    )

    print(
        f"MLflow tracking URI: "
        f"{tracking_uri}"
    )

    print(
        f"MLflow registry URI: "
        f"{tracking_uri}"
    )

    return tracking_uri


def train(config):
    """
    Main training pipeline.

    Pipeline:
        Configuration
        -> Dataset
        -> Training
        -> Validation
        -> Best checkpoint
        -> Test evaluation
        -> MLflow tracking
        -> Model logging
        -> Model registry
    """

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    seed = config.get(
        "project",
        {},
    ).get(
        "seed",
        42,
    )

    torch.manual_seed(seed)

    # ---------------------------------------------------------
    # MLflow server configuration
    # ---------------------------------------------------------

    tracking_uri = configure_mlflow()

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = get_device()

    print(
        f"Using device: {device}"
    )

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    train_loader, val_loader, test_loader = (
        get_dataloaders(config)
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = create_model(
        config
    ).to(device)

    # ---------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------

    epochs = config[
        "training"
    ]["epochs"]

    learning_rate = config[
        "training"
    ]["learning_rate"]

    batch_size = config[
        "training"
    ]["batch_size"]

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # ---------------------------------------------------------
    # MLflow experiment
    # ---------------------------------------------------------

    experiment_name = config[
        "mlflow"
    ]["experiment_name"]

    mlflow.set_experiment(
        experiment_name
    )

    print(
        f"MLflow experiment: "
        f"{experiment_name}"
    )

    # ---------------------------------------------------------
    # Start MLflow run
    # ---------------------------------------------------------

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        print(
            f"MLflow Run ID: "
            f"{run_id}"
        )

        # -----------------------------------------------------
        # Log parameters
        # -----------------------------------------------------

        mlflow.log_params(
            {
                "model_name":
                    config["model"]["name"],

                "epochs":
                    epochs,

                "batch_size":
                    batch_size,

                "learning_rate":
                    learning_rate,

                "dataset":
                    config["data"]["dataset"],

                "train_split":
                    config["data"]["train_split"],

                "device":
                    str(device),

                "seed":
                    seed,

                "tracking_uri":
                    tracking_uri,
            }
        )

        # -----------------------------------------------------
        # Artifact directory
        # -----------------------------------------------------

        artifacts_dir = Path(
            "artifacts"
        )

        artifacts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        best_model_path = (
            artifacts_dir
            / "best_model.pth"
        )

        # -----------------------------------------------------
        # Training state
        # -----------------------------------------------------

        best_val_accuracy = 0.0
        best_epoch = 0

        # -----------------------------------------------------
        # Training loop
        # -----------------------------------------------------

        for epoch in range(
            1,
            epochs + 1,
        ):

            model.train()

            running_loss = 0.0
            total_samples = 0

            for (
                images,
                labels,
            ) in train_loader:

                images = images.to(
                    device
                )

                labels = labels.to(
                    device
                )

                # Clear gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(
                    images
                )

                # Loss
                loss = criterion(
                    outputs,
                    labels,
                )

                # Backpropagation
                loss.backward()

                # Update model
                optimizer.step()

                running_loss += (
                    loss.item()
                    * images.size(0)
                )

                total_samples += (
                    images.size(0)
                )

            train_loss = (
                running_loss
                / total_samples
            )

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            val_metrics = evaluate_model(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
            )

            val_loss = (
                val_metrics["loss"]
            )

            val_accuracy = (
                val_metrics["accuracy"]
            )

            # -------------------------------------------------
            # Log epoch metrics
            # -------------------------------------------------

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_accuracy",
                val_accuracy,
                step=epoch,
            )

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"| Train Loss: "
                f"{train_loss:.4f} "
                f"| Val Loss: "
                f"{val_loss:.4f} "
                f"| Val Accuracy: "
                f"{val_accuracy:.4f}"
            )

            # -------------------------------------------------
            # Save best checkpoint
            # -------------------------------------------------

            if (
                val_accuracy
                > best_val_accuracy
            ):

                best_val_accuracy = (
                    val_accuracy
                )

                best_epoch = epoch

                torch.save(
                    model.state_dict(),
                    best_model_path,
                )

                print(
                    f"Best model updated "
                    f"at epoch {epoch} "
                    f"with validation accuracy "
                    f"{val_accuracy:.4f}"
                )

        # -----------------------------------------------------
        # Reload best checkpoint
        # -----------------------------------------------------

        model.load_state_dict(
            torch.load(
                best_model_path,
                map_location=device,
            )
        )

        model.to(device)

        model.eval()

        print(
            f"Reloaded best model "
            f"from epoch {best_epoch} "
            f"with validation accuracy "
            f"{best_val_accuracy:.4f}"
        )

        # -----------------------------------------------------
        # Final test evaluation
        # -----------------------------------------------------

        test_metrics = evaluate_model(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        test_loss = (
            test_metrics["loss"]
        )

        test_accuracy = (
            test_metrics["accuracy"]
        )

        print(
            f"Test Loss: "
            f"{test_loss:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{test_accuracy:.4f}"
        )

        # -----------------------------------------------------
        # Log final metrics
        # -----------------------------------------------------

        mlflow.log_metric(
            "test_loss",
            test_loss,
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy,
        )

        mlflow.log_metric(
            "best_val_accuracy",
            best_val_accuracy,
        )

        mlflow.log_metric(
            "best_epoch",
            best_epoch,
        )

        # -----------------------------------------------------
        # Log raw checkpoint
        # -----------------------------------------------------

        mlflow.log_artifact(
            str(best_model_path),
            artifact_path="checkpoints",
        )

        # -----------------------------------------------------
        # Prepare complete MLflow model
        # -----------------------------------------------------

        model_cpu = model.to(
            "cpu"
        )

        model_cpu.eval()

        input_example = torch.randn(
            1,
            1,
            28,
            28,
        )

        with torch.no_grad():

            example_output = (
                model_cpu(
                    input_example
                )
            )

        # -----------------------------------------------------
        # Infer model signature
        # -----------------------------------------------------

        signature = infer_signature(
            input_example.numpy(),
            example_output.numpy(),
        )

        # -----------------------------------------------------
        # Log complete PyTorch model
        # -----------------------------------------------------

        model_info = (
            mlflow.pytorch.log_model(
                pytorch_model=model_cpu,

                name="pytorch-model",

                input_example=(
                    input_example.numpy()
                ),

                signature=signature,

                serialization_format=(
                    "pickle"
                ),
            )
        )

        print(
            f"MLflow model logged: "
            f"{model_info.model_uri}"
        )

        # -----------------------------------------------------
        # Register model
        # -----------------------------------------------------

        registry_name = config.get(
            "mlflow",
            {},
        ).get(
            "registered_model_name",
            "industrial-baseline-cnn",
        )

        registered_model = (
            mlflow.register_model(
                model_uri=(
                    model_info.model_uri
                ),
                name=registry_name,
            )
        )

        registered_version = (
            registered_model.version
        )

        print(
            f"Registered model: "
            f"{registry_name}"
        )

        print(
            f"Registered model version: "
            f"{registered_version}"
        )

        # -----------------------------------------------------
        # Final summary
        # -----------------------------------------------------

        print(
            "\n"
            "======================================"
        )

        print(
            "Training completed successfully"
        )

        print(
            f"Tracking server: "
            f"{tracking_uri}"
        )

        print(
            f"Experiment: "
            f"{experiment_name}"
        )

        print(
            f"Run ID: "
            f"{run_id}"
        )

        print(
            f"Best epoch: "
            f"{best_epoch}"
        )

        print(
            f"Best validation accuracy: "
            f"{best_val_accuracy:.4f}"
        )

        print(
            f"Final test accuracy: "
            f"{test_accuracy:.4f}"
        )

        print(
            f"Registered model: "
            f"{registry_name}"
        )

        print(
            f"Model version: "
            f"{registered_version}"
        )

        print(
            "======================================"
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train, evaluate, track, "
            "log and register a PyTorch "
            "model using a YAML configuration."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to YAML "
            "configuration file."
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    train(config)


if __name__ == "__main__":
    main()