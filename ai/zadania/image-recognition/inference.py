import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from feed_forward import FeedForwardNetwork


def load_image_to_mnist_vector(image_path: str, target_size=(28, 28), invert=True):
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    arr = np.array(img, dtype=np.float64)
    if invert:
        arr = 255.0 - arr

    arr = arr.reshape(-1) / 255.0
    arr = arr.astype(np.float64)
    return arr


def parse_args():
    parser = argparse.ArgumentParser(description='Run MNIST inference on a PNG image using a saved model.')
    parser.add_argument('--model-path', type=str, help='Path to the saved model (.npy)')
    parser.add_argument('--image-path', type=str, help='Path to the input PNG image')
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    model_path = Path(args.model_path)

    if not image_path.exists():
        raise FileNotFoundError(f'Obrázok nebol nájdený: {image_path}')
    if not model_path.exists():
        raise FileNotFoundError(f'Model nebol nájdený: {model_path}')

    x = load_image_to_mnist_vector(str(image_path), target_size=(28, 28), invert=True)
    x = x.reshape(1, -1)

    model = FeedForwardNetwork.load(str(model_path))
    probs = model.predict_proba(x)
    prediction = int(model.predict(x)[0])
    confidence = float(np.max(probs[0]))

    print(f'Predikcia: {prediction}')
    print(f'Pevnosť: {confidence:.4f}')
    print(f'Rozdelenie pravdepodobností: {np.round(probs[0], 4)}')


if __name__ == '__main__':
    main()
