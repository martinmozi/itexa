import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from feed_forward import FeedForwardNetwork


def _shift(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)] = \
        a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def normalize_like_mnist(arr, threshold=0.1):
    """Replikuje pôvodné MNIST predspracovanie.

    Číslicu oreže na bounding box, zmenší tak, aby sa zmestila do 20x20 boxu
    pri zachovaní pomeru strán, vloží ju do 28x28 a posunie tak, aby jej
    ťažisko sedelo v strede.

    arr: 2D float 0..1, biela číslica na čiernom pozadí.
    """
    ys, xs = np.nonzero(arr > threshold)
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.float64)

    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    small = np.asarray(
        Image.fromarray((crop * 255.0).astype(np.uint8))
             .resize((new_w, new_h), Image.Resampling.LANCZOS),
        dtype=np.float64,
    ) / 255.0

    canvas = np.zeros((28, 28), dtype=np.float64)
    y0 = (28 - new_h) // 2
    x0 = (28 - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = small

    total = canvas.sum()
    if total > 0:
        yy, xx = np.mgrid[0:28, 0:28]
        dy = int(round(13.5 - (yy * canvas).sum() / total))
        dx = int(round(13.5 - (xx * canvas).sum() / total))
        canvas = _shift(canvas, dy, dx)

    return canvas


def load_image_to_mnist_vector(image_path: str, invert=True, normalize=True):
    img = Image.open(image_path).convert('L')

    arr = np.array(img, dtype=np.float64)
    if invert:
        arr = 255.0 - arr
    arr /= 255.0

    if normalize:
        arr = normalize_like_mnist(arr)
    elif arr.shape != (28, 28):
        arr = np.asarray(
            Image.fromarray((arr * 255.0).astype(np.uint8))
                 .resize((28, 28), Image.Resampling.LANCZOS),
            dtype=np.float64,
        ) / 255.0

    return arr.reshape(-1)


def parse_args():
    parser = argparse.ArgumentParser(description='Run MNIST inference on a PNG image using a saved model.')
    parser.add_argument('--model-path', type=str, help='Path to the saved model (.npy)')
    parser.add_argument('--image-path', type=str, help='Path to the input PNG image')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Skip the MNIST-style crop/rescale/centering step')
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    model_path = Path(args.model_path)

    if not image_path.exists():
        raise FileNotFoundError(f'Obrázok nebol nájdený: {image_path}')
    if not model_path.exists():
        raise FileNotFoundError(f'Model nebol nájdený: {model_path}')

    x = load_image_to_mnist_vector(str(image_path), invert=True, normalize=not args.no_normalize)
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
