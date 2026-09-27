import argparse

from mlflow import MlflowClient


REGISTERED_MODEL_NAME = "industrial-baseline-cnn"


def promote_model(version: int, alias: str = "champion"):
    """
    Assign an MLflow alias to a registered model version.

    Example:
        Version 1 -> champion
    """

    client = MlflowClient()

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=alias,
        version=str(version),
    )

    print(
        "\n======================================"
    )

    print("Model promotion successful")

    print(
        f"Model:   {REGISTERED_MODEL_NAME}"
    )

    print(
        f"Version: {version}"
    )

    print(
        f"Alias:   {alias}"
    )

    print(
        "======================================"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Promote a registered MLflow "
            "model version using an alias."
        )
    )

    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Registered model version.",
    )

    parser.add_argument(
        "--alias",
        type=str,
        default="champion",
        help=(
            "Alias to assign. "
            "Default: champion"
        ),
    )

    args = parser.parse_args()

    promote_model(
        version=args.version,
        alias=args.alias,
    )


if __name__ == "__main__":
    main()