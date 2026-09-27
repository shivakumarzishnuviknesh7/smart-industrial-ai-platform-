import torch

from src.model import BaselineCNN


def test_model_output_shape():
    """
    Verify that the CNN accepts a FashionMNIST image
    and produces logits for 10 classes.
    """

    model = BaselineCNN(
        num_classes=10
    )

    model.eval()

    dummy_input = torch.randn(
        2,
        1,
        28,
        28,
    )

    with torch.no_grad():
        output = model(
            dummy_input
        )

    assert output.shape == (
        2,
        10,
    )


def test_model_single_prediction():
    """
    Verify inference for one image.
    """

    model = BaselineCNN(
        num_classes=10
    )

    model.eval()

    dummy_input = torch.randn(
        1,
        1,
        28,
        28,
    )

    with torch.no_grad():
        output = model(
            dummy_input
        )

    predicted_class = (
        output.argmax(
            dim=1
        ).item()
    )

    assert 0 <= predicted_class <= 9