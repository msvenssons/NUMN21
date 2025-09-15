import numpy as np
import tqdm
import matplotlib.pyplot as plt

# implemented this so the API is similar to PyTorch; a Sequential class that stacks layers and activation functions

class Layer:
    """
    A fully connected neural network layer supporting forward and backward passes.
    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        requires_grad (bool, optional): If True, gradients are computed during backpropagation. Defaults to True.
    """
    def __init__(self, input_size: int, output_size: int, requires_grad=True):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0/input_size)
        self.b = np.random.uniform(0, 0.5, (1, output_size))
        self.requires_grad = requires_grad
        self.cache = None
        self.dW = None
        self.db = None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        try: 
            # might have to transpose weights depending on input shape
            out = x @ self.weights + self.b
            # save relevant values for backpropagation
            self.cache = x
        except Exception as e:
            print(f"Error in forward pass (check input dimensions): {e}")
        return out
    
    def _backward(self, dY: np.ndarray) -> np.ndarray:
        if self.requires_grad:
            x = self.cache
            batch_size = x.shape[0]
            self.dW = x.T @ dY / batch_size
            self.db = np.sum(dY, axis=0, keepdims=True) / batch_size
            dY = dY @ self.weights.T 
        return dY



# ---------------------------------------------------------------------------------------------------


class ReLU:
    """ReLU activation function: used as a layer in Sequential model."""
    def __init__(self):
        self.cache = None
        self.dbW = None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.maximum(0, x)
        mask = np.where(x > 0, 1, 0)
        self.cache = (mask, out)
        return out
    
    def _backward(self, dY: np.ndarray) -> np.ndarray:
        return dY * self.cache[0]


# ---------------------------------------------------------------------------------------------------


class Sigmoid:
    """Sigmoid activation function: used as a layer in Sequential model."""
    def __init__(self):
        self.cache = None
        self.dbW = None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # clip large values to avoid overflow
        x = np.clip(x, -500, 500)
        out = 1 / (1 + np.exp(-x))
        self.cache = out
        return out
    
    def _backward(self, dY: np.ndarray) -> np.ndarray:
        out = self.cache
        return dY * (out * (1 - out))


# ---------------------------------------------------------------------------------------------------


class Sequential:
    """Sequential model to stack layers and activation functions into a feedforward neural network with forward and backward passes."""
    def __init__(self, *layers):
        self.layers = layers
        self.out = None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return x
    
    def _backward(self, labels: np.ndarray):
        dY = self._squared_error_grad(self.out, labels)
        for layer in reversed(self.layers):
            dY = layer._backward(dY)
        return 0
    
    def _squared_error_grad(self, preds: np.ndarray, labels: np.ndarray) -> float:
        return preds - labels
    
    def _mse_loss(self, preds: np.ndarray, labels: np.ndarray) -> float:
        return (1/2)*np.mean((preds - labels) ** 2)
    
    def _gradient_step(self, lr: float = 0.01):
        for layer in self.layers:
            if isinstance(layer, Layer) and layer.requires_grad:
                layer.weights -= lr * layer.dW
                layer.b -= lr * layer.db
                layer.dW = None
                layer.db = None
                layer.cache = None
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray = None, y_val: np.ndarray = None, epochs: int = 10, batch_size: int = 32, lr: float = 0.01):
        """
        Simple training function that uses SGD to train the model. Prints training and validation loss.

        Args:
            x_train (np.ndarray): Training input data of shape (num_samples, num_features).
            y_train (np.ndarray): Training target data of shape (num_samples, 1).
            x_val (np.ndarray, optional): Validation input data of shape (num_val_samples, num_features).
            y_val (np.ndarray, optional): Validation target data of shape (num_val_samples, 1).
            epochs (int, optional): Number of training epochs. Defaults to 10.
            batch_size (int, optional): Size of each mini-batch for SGD. Defaults to 32.
            lr (float, optional): Learning rate for SGD. Defaults to 0.01."""

        sample_size = x_train.shape[0]
        pbar = tqdm.tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            perm = np.random.permutation(sample_size)
            x_perm = x_train[perm]
            y_perm = y_train[perm]
            for i in range(0, sample_size, batch_size):
                x_batch = x_perm[i:i+batch_size]
                y_batch = y_perm[i:i+batch_size]
                self(x_batch)
                self._backward(y_batch)
                self._gradient_step(lr)
            full_train_pred = self(x_train)
            train_loss = self._mse_loss(full_train_pred, y_train)
            if x_val is not None and y_val is not None:
                full_val_pred = self(x_val)
                val_loss = self._mse_loss(full_val_pred, y_val)
                pbar.set_description(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                pbar.set_description(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
            pbar.refresh()
