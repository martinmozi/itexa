import argparse
import os
from dataclasses import dataclass, field
from os.path import join
from loader import MnistDataloader
import numpy as np

try:
    import matplotlib
    matplotlib.use("Qt5Agg", force=True)
except Exception:
    try:
        import matplotlib
        matplotlib.use("TkAgg", force=True)
    except Exception:
        import matplotlib
        matplotlib.use("Agg", force=True)


@dataclass
class NetworkConfig:
    input_dim: int = 784
    hidden_layers: list[int] = field(default_factory=lambda: [128, 64])
    output_dim: int = 10
    activation: str = "relu"
    seed: int = 42


@dataclass
class AdamConfig:
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


class FeedForwardNetwork:
    def __init__(self, config: NetworkConfig, adam: AdamConfig):
        self.config = config
        self.adam = adam
        self.rng = np.random.default_rng(config.seed)
        self.activation_name = (config.activation or 'relu').lower()
        self.activation_fn, self.activation_grad_fn = self._select_activation(self.activation_name)

        layer_sizes = [config.input_dim, *config.hidden_layers, config.output_dim]
        self.weights = []
        self.biases = []
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []

        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            if self.activation_name in {"relu", "leaky_relu"}:
                scale = np.sqrt(2.0 / in_size)
            else:
                scale = np.sqrt(1.0 / in_size)

            W = self.rng.normal(0.0, scale, size=(in_size, out_size)).astype(np.float64)
            b = np.zeros(out_size, dtype=np.float64)

            self.weights.append(W)
            self.biases.append(b)
            self.m_weights.append(np.zeros_like(W))
            self.v_weights.append(np.zeros_like(W))
            self.m_biases.append(np.zeros_like(b))
            self.v_biases.append(np.zeros_like(b))

        self._step = 0

    @staticmethod
    def _select_activation(name):
        if name == "relu":
            return FeedForwardNetwork._relu, FeedForwardNetwork._relu_grad
        if name == "leaky_relu":
            return FeedForwardNetwork._leaky_relu, FeedForwardNetwork._leaky_relu_grad
        raise ValueError(f"Unsupported activation '{name}'. Supported: relu, leaky_relu")

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _leaky_relu(x, alpha=0.01):
        return np.where(x > 0.0, x, alpha * x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0).astype(np.float64)

    @staticmethod
    def _leaky_relu_grad(x, alpha=0.01):
        grad = np.ones_like(x)
        grad[x < 0] = alpha
        return grad.astype(np.float64)

    def _softmax(self, logits):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def _forward(self, X):
        activations = [X]
        pre_activations = []

        for idx, W in enumerate(self.weights[:-1]):
            z = activations[-1] @ W + self.biases[idx]
            pre_activations.append(z)
            a = self.activation_fn(z)
            activations.append(a)

        logits = activations[-1] @ self.weights[-1] + self.biases[-1]
        pre_activations.append(logits)
        probs = self._softmax(logits)

        return activations, pre_activations, probs

    def _loss_and_gradients(self, X, y):
        activations, pre_activations, probs = self._forward(X)
        n_samples = X.shape[0]

        loss = -np.log(probs[np.arange(n_samples), y] + 1e-12).mean()

        d_logits = probs.copy()
        d_logits[np.arange(n_samples), y] -= 1.0
        d_logits /= n_samples

        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        d_a = d_logits
        for layer_index in reversed(range(len(self.weights))):
            W = self.weights[layer_index]
            if layer_index == len(self.weights) - 1:
                d_w = activations[layer_index].T @ d_a
                d_b = d_a.sum(axis=0)
                d_prev = d_a @ W.T
            else:
                d_z = d_prev * self.activation_grad_fn(pre_activations[layer_index])
                d_w = activations[layer_index].T @ d_z
                d_b = d_z.sum(axis=0)
                d_prev = d_z @ W.T

            grads_w[layer_index] = d_w
            grads_b[layer_index] = d_b

            if layer_index > 0:
                d_a = d_prev

        return loss, list(zip(grads_w, grads_b))

    def _apply_adam(self, grads):
        self._step += 1

        for idx, (grad_w, grad_b) in enumerate(grads):
            self.m_weights[idx] = (
                self.adam.beta1 * self.m_weights[idx] + (1.0 - self.adam.beta1) * grad_w
            )
            self.v_weights[idx] = (
                self.adam.beta2 * self.v_weights[idx] + (1.0 - self.adam.beta2) * (grad_w ** 2)
            )
            self.m_biases[idx] = (
                self.adam.beta1 * self.m_biases[idx] + (1.0 - self.adam.beta1) * grad_b
            )
            self.v_biases[idx] = (
                self.adam.beta2 * self.v_biases[idx] + (1.0 - self.adam.beta2) * (grad_b ** 2)
            )

            m_w_hat = self.m_weights[idx] / (1.0 - self.adam.beta1 ** self._step)
            v_w_hat = self.v_weights[idx] / (1.0 - self.adam.beta2 ** self._step)
            m_b_hat = self.m_biases[idx] / (1.0 - self.adam.beta1 ** self._step)
            v_b_hat = self.v_biases[idx] / (1.0 - self.adam.beta2 ** self._step)

            self.weights[idx] -= self.adam.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.adam.epsilon)
            self.biases[idx] -= self.adam.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.adam.epsilon)

    def fit(self, X, y, epochs=10, batch_size=64, verbose=True, validation_split=0.1):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples = X.shape[0]

        if 0.0 < validation_split < 1.0:
            val_size = max(1, int(round(n_samples * validation_split)))
            if val_size >= n_samples:
                val_size = n_samples // 2
            indices = np.random.permutation(n_samples)
            val_idx = indices[:val_size]
            train_idx = indices[val_size:]
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
        else:
            X_train = X
            y_train = y
            X_val = None
            y_val = None

        history = []
        acc_history = []
        val_history = []

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(len(X_train))
            epoch_loss = 0.0

            for start in range(0, len(X_train), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_x = X_train[batch_idx]
                batch_y = y_train[batch_idx]

                loss, grads = self._loss_and_gradients(batch_x, batch_y)
                self._apply_adam(grads)
                epoch_loss += loss * batch_x.shape[0]

            epoch_loss /= len(X_train)
            history.append(epoch_loss)

            train_accuracy = self.accuracy(X_train, y_train)
            acc_history.append(train_accuracy)

            if X_val is not None and y_val is not None:
                val_accuracy = self.accuracy(X_val, y_val)
                val_history.append(val_accuracy)
            else:
                val_history.append(None)

            if verbose:
                if X_val is not None and y_val is not None:
                    print(
                        f"Epoch {epoch:02d}/{epochs}: loss={epoch_loss:.4f}, "
                        f"train_acc={train_accuracy:.4f}, val_acc={val_accuracy:.4f}"
                    )
                else:
                    print(f"Epoch {epoch:02d}/{epochs}: loss={epoch_loss:.4f}, acc={train_accuracy:.4f}")

        self.loss_history = history
        self.accuracy_history = acc_history
        self.val_accuracy_history = [v for v in val_history if v is not None]

        if self.val_accuracy_history:
            best_val_accuracy = max(self.val_accuracy_history)
            last_val_accuracy = self.val_accuracy_history[-1]
            gap = self.accuracy_history[-1] - last_val_accuracy
            if last_val_accuracy < best_val_accuracy - 0.02:
                print(
                    f"Varovanie: overfitting detekovaný (best_val={best_val_accuracy:.4f}, "
                    f"last_val={last_val_accuracy:.4f}, train_val_gap={gap:.4f})"
                )

        return history, acc_history

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        _, _, probs = self._forward(X)
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def save(self, path):
        state = {
            'config': {
                'input_dim': self.config.input_dim,
                'hidden_layers': self.config.hidden_layers,
                'output_dim': self.config.output_dim,
                'activation': self.config.activation,
                'seed': self.config.seed,
            },
            'adam': {
                'learning_rate': self.adam.learning_rate,
                'beta1': self.adam.beta1,
                'beta2': self.adam.beta2,
                'epsilon': self.adam.epsilon,
            },
            'step': self._step,
            'weights': [w.copy() for w in self.weights],
            'biases': [b.copy() for b in self.biases],
            'm_weights': [m.copy() for m in self.m_weights],
            'v_weights': [v.copy() for v in self.v_weights],
            'm_biases': [m.copy() for m in self.m_biases],
            'v_biases': [v.copy() for v in self.v_biases],
            'loss_history': getattr(self, 'loss_history', []),
            'accuracy_history': getattr(self, 'accuracy_history', []),
        }
        np.save(path, state, allow_pickle=True)

        model_dir = os.path.dirname(path) or '.'
        base_name = os.path.splitext(os.path.basename(path))[0]
        svg_path = os.path.join(model_dir, f'{base_name}.svg')
        self.save_architecture_svg(svg_path)
        return svg_path

    @classmethod
    def load(cls, path):
        state = np.load(path, allow_pickle=True).item()

        config = NetworkConfig(
            input_dim=state['config']['input_dim'],
            hidden_layers=state['config']['hidden_layers'],
            output_dim=state['config']['output_dim'],
            activation=state['config']['activation'],
            seed=state['config']['seed'],
        )
        adam = AdamConfig(
            learning_rate=state['adam']['learning_rate'],
            beta1=state['adam']['beta1'],
            beta2=state['adam']['beta2'],
            epsilon=state['adam']['epsilon'],
        )

        model = cls(config=config, adam=adam)
        model._step = state['step']
        model.weights = [w.copy() for w in state['weights']]
        model.biases = [b.copy() for b in state['biases']]
        model.m_weights = [m.copy() for m in state['m_weights']]
        model.v_weights = [v.copy() for v in state['v_weights']]
        model.m_biases = [m.copy() for m in state['m_biases']]
        model.v_biases = [v.copy() for v in state['v_biases']]
        model.loss_history = state.get('loss_history', [])
        model.accuracy_history = state.get('accuracy_history', [])
        return model

    def save_architecture_svg(self, output_path='feed_forward_default.svg'):
        layers = [self.config.input_dim, *self.config.hidden_layers, self.config.output_dim]
        layer_width = 170
        layer_height = 34
        x_spacing = 220
        left_margin = 40
        top_margin = 70
        total_width = left_margin + (len(layers) - 1) * x_spacing + layer_width + 40
        total_height = 220

        lines = []
        lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}">')
        lines.append('<defs><style>')
        lines.append('rect { fill: #f7f7f7; stroke: #333; stroke-width: 2; }')
        lines.append('text { font-family: Arial, sans-serif; fill: #111; text-anchor: middle; dominant-baseline: middle; }')
        lines.append('.edge { stroke: #4f81bd; stroke-width: 2; fill: none; opacity: 0.8; }')
        lines.append('</style></defs>')

        for i, size in enumerate(layers):
            x = left_margin + i * x_spacing
            y = top_margin
            lines.append(f'<rect x="{x}" y="{y}" width="{layer_width}" height="{layer_height}" rx="8" ry="8"/>')
            label = f'{size}'
            lines.append(f'<text x="{x + layer_width / 2}" y="{y + layer_height / 2}" font-size="18">{label}</text>')

            if i < len(layers) - 1:
                next_x = left_margin + (i + 1) * x_spacing
                lines.append(f'<line class="edge" x1="{x + layer_width}" y1="{y + layer_height / 2}" x2="{next_x}" y2="{top_margin + layer_height / 2}" />')

        title = f'FeedForwardNetwork [{" -> ".join(str(v) for v in layers)}]'
        lines.append(f'<text x="{total_width / 2}" y="30" font-size="20" font-weight="bold" text-anchor="middle">{title}</text>')
        lines.append('</svg>')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f'Schéma siete bolo uložené do {output_path}')


def prepare_mnist_data():
    input_path = './mnist'
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    x_train = np.asarray(x_train, dtype=np.float64).reshape(-1, 28 * 28) / 255.0
    x_test = np.asarray(x_test, dtype=np.float64).reshape(-1, 28 * 28) / 255.0
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    return x_train, y_train, x_test, y_test


def parse_args():
    parser = argparse.ArgumentParser(description='Train a feed-forward neural network on MNIST.')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[128, 64],
                        help='Hidden layer sizes, e.g. --hidden-layers 128 64 32')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'leaky_relu'], help='Hidden-layer activation function')
    parser.add_argument('--validation-split', type=float, default=0.1, help='Fraction of training data used as validation during training; set 0.0 to disable.')
    parser.add_argument('--save-model', type=str, default=None, help='Path to save trained model (.npy)')
    return parser.parse_args()


def main():
    args = parse_args()
    x_train, y_train, x_test, y_test = prepare_mnist_data()

    config = NetworkConfig(
        input_dim=28 * 28,
        hidden_layers=args.hidden_layers,
        output_dim=10,
        activation=args.activation,
        seed=args.seed,
    )
    adam = AdamConfig(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
    )

    model = FeedForwardNetwork(config=config, adam=adam)
    history, accuracy_history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
        validation_split=args.validation_split,
    )

    train_accuracy = model.accuracy(x_train, y_train)
    test_accuracy = model.accuracy(x_test, y_test)

    print(f"\nFinal training accuracy: {train_accuracy:.4f}")
    print(f"Final test accuracy:     {test_accuracy:.4f}")
    if getattr(model, 'val_accuracy_history', None):
        best_val_accuracy = max(model.val_accuracy_history)
        final_val_accuracy = model.val_accuracy_history[-1]
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        print(f"Final validation accuracy: {final_val_accuracy:.4f}")
    if history:
        print(f"Loss history: {history[-5:]}")

    if args.save_model:
        svg_path = model.save(args.save_model)
        print(f'Model uložený do {args.save_model}')
        print(f'Schéma siete uložená do {svg_path}')


if __name__ == '__main__':
    main()
