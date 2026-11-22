import argparse
import os
import subprocess
import sys

import requests


REGISTERED_MODEL_NAME = "industrial-baseline-cnn"
MODEL_ALIAS = "champion"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5001",
).rstrip("/")


def run_command(command):
    """
    Run a shell command and return its stdout.

    Exit immediately if the command fails.
    """
    result = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print(result.stderr.strip())
        sys.exit(result.returncode)

    return result.stdout.strip()


def mlflow_get(path, params=None):
    """
    Send a GET request to the MLflow REST API.
    """
    url = f"{MLFLOW_TRACKING_URI}{path}"

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        print(
            f"\nMLflow request failed:\n{exc}"
        )
        sys.exit(1)


def mlflow_post(path, payload):
    """
    Send a POST request to the MLflow REST API.
    """
    url = f"{MLFLOW_TRACKING_URI}{path}"

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()

        if response.content:
            return response.json()

        return {}

    except requests.RequestException as exc:
        print(
            f"\nMLflow request failed:\n{exc}"
        )
        sys.exit(1)


def get_champion_version():
    """
    Read the registered model version currently assigned
    to the MLflow champion alias.
    """
    response = mlflow_get(
        "/api/2.0/mlflow/registered-models/alias",
        params={
            "name": REGISTERED_MODEL_NAME,
            "alias": MODEL_ALIAS,
        },
    )

    model_version = response.get("model_version")

    if not model_version:
        print(
            f"\nAlias '{MODEL_ALIAS}' was not found "
            f"for model '{REGISTERED_MODEL_NAME}'."
        )
        sys.exit(1)

    return model_version["version"]


def show_status():
    """
    Show Kubernetes deployment status and the current
    MLflow champion model version.
    """
    print(
        "\n=== Smart Industrial AI Platform Status ===\n"
    )

    image = run_command(
        "kubectl get deployment "
        "smart-industrial-ai-platform "
        "-o jsonpath="
        "'{.spec.template.spec.containers[0].image}'"
    )

    pods = run_command(
        "kubectl get pods "
        "-l app=smart-industrial-ai-platform "
        "-o wide"
    )

    champion_version = get_champion_version()

    print(
        f"Application image: {image}"
    )

    print(
        f"Champion model version: "
        f"{champion_version}"
    )

    print("\nKubernetes pods:")
    print(pods)


def promote_model(version):
    """
    Move the MLflow champion alias to the specified
    registered model version.
    """
    payload = {
        "name": REGISTERED_MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(version),
    }

    mlflow_post(
        "/api/2.0/mlflow/registered-models/alias",
        payload,
    )

    print(
        f"\nModel version {version} promoted "
        f"to alias '{MODEL_ALIAS}'."
    )


def rollback_application(image_tag):
    """
    Update the GitOps deployment YAML to use another
    Docker image tag.

    Flux will deploy the change after it is committed
    and pushed to Git.
    """
    deployment_file = (
        "clusters/docker-desktop/apps/"
        "smart-industrial-ai-platform/"
        "deployment.yaml"
    )

    image_line = run_command(
        f"grep 'image:' {deployment_file}"
    )

    current_image = (
        image_line
        .split("image:", 1)[1]
        .strip()
    )

    registry_path = current_image.rsplit(
        ":",
        1,
    )[0]

    new_image = (
        f"{registry_path}:{image_tag}"
    )

    if current_image == new_image:
        print(
            "\nRequested image is already present "
            "in the GitOps configuration."
        )
        return

    run_command(
        f"sed -i '' "
        f"'s|image: {current_image}|"
        f"image: {new_image}|' "
        f"{deployment_file}"
    )

    print("\nApplication image updated:")
    print(current_image)
    print("↓")
    print(new_image)

    print(
        "\nNext steps:"
        "\n1. Commit deployment.yaml"
        "\n2. Push to GitLab"
        "\n3. Flux reconciles Kubernetes"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Self-service CLI for the "
            "Smart Industrial AI Platform."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "status",
        help=(
            "Show application image, "
            "champion model version, "
            "and Kubernetes pod status."
        ),
    )

    promote_parser = subparsers.add_parser(
        "promote",
        help=(
            "Promote an MLflow model version "
            "to the champion alias."
        ),
    )

    promote_parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="MLflow registered model version.",
    )

    rollback_parser = subparsers.add_parser(
        "rollback",
        help=(
            "Update the GitOps deployment "
            "to a Docker image tag."
        ),
    )

    rollback_parser.add_argument(
        "--image-tag",
        required=True,
        help="Docker image tag.",
    )

    args = parser.parse_args()

    if args.command == "status":
        show_status()

    elif args.command == "promote":
        promote_model(
            args.version
        )

    elif args.command == "rollback":
        rollback_application(
            args.image_tag
        )


if __name__ == "__main__":
    main()