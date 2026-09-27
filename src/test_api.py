import requests
from torchvision import datasets, transforms


API_URL = "http://127.0.0.1:8000/predict"


CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


def main():
    # ---------------------------------------------------------
    # Load FashionMNIST test dataset
    # ---------------------------------------------------------

    dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    # ---------------------------------------------------------
    # Pick one test sample
    # ---------------------------------------------------------

    image, label = dataset[0]

    actual_label = CLASS_NAMES[label]

    # image shape:
    # [1, 28, 28]

    # Convert from [0, 1]
    # back to [0, 255]
    image = image.squeeze(0)

    pixels = (
        image * 255.0
    ).tolist()

    # ---------------------------------------------------------
    # Build API request
    # ---------------------------------------------------------

    payload = {
        "pixels": pixels
    }

    print(
        f"Sending test image to: "
        f"{API_URL}"
    )

    print(
        f"Actual class: "
        f"{label}"
    )

    print(
        f"Actual label: "
        f"{actual_label}"
    )

    # ---------------------------------------------------------
    # Call FastAPI
    # ---------------------------------------------------------

    response = requests.post(
        API_URL,
        json=payload,
        timeout=30,
    )

    # ---------------------------------------------------------
    # Check response
    # ---------------------------------------------------------

    if response.status_code != 200:

        print(
            "Prediction request failed"
        )

        print(
            f"Status code: "
            f"{response.status_code}"
        )

        print(
            f"Response: "
            f"{response.text}"
        )

        return

    result = response.json()

    # ---------------------------------------------------------
    # Print prediction result
    # ---------------------------------------------------------

    print(
        "\n"
        "======================================"
    )

    print(
        "Prediction successful"
    )

    print(
        f"Actual label: "
        f"{actual_label}"
    )

    print(
        f"Predicted label: "
        f"{result['predicted_label']}"
    )

    print(
        f"Predicted class: "
        f"{result['predicted_class']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.4f}"
    )

    print(
        f"Model: "
        f"{result['model_name']}"
    )

    print(
        f"Alias: "
        f"{result['model_alias']}"
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()