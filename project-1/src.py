import numpy as np


class Layer:
    """A fully connected layer with forward and backward passes for Sequential model"""
    def __init__(self, input_size: int, output_size: int, requires_grad=True):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0/input_size)
        self.b = np.random.uniform(-1, 1, (1, output_size))
        self.requires_grad = requires_grad
        self.cache = None
        self.dW = None
        self.db = None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        try: 
            # might have to transpose weights depending on input shape
            out = x @ self.weights + self.b
            # save relevant values for backpropagation
            self.cache = (x, out)
        except Exception as e:
            print(f"Error in forward pass (check input dimensions): {e}")
        return out
    
    def _backward(self, dY: np.ndarray) -> np.ndarray:
        if self.requires_grad:
            x, out = self.cache
            self.dW = x.T @ dY
            self.db = np.sum(dY, axis=0, keepdims=True)
            dY = dY @ self.weights.T
        return dY



# ---------------------------------------------------------------------------------------------------


class ReLU:
    """ReLU activation function: used as a layer in Sequential model"""
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
    """Sigmoid activation function: used as a layer in Sequential model"""
    def __init__(self):
        self.cache = None
        self.dbW = None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = 1 / (1 + np.exp(-x))
        self.cache = (x, out)
        return out
    
    def _backward(self, dY: np.ndarray) -> np.ndarray:
        x, out = self.cache
        dY = dY * (out * (1 - out))
        return dY


# ---------------------------------------------------------------------------------------------------


class Sequential:
    """Sequential model to stack layers and activation functions into a feedforward neural network."""
    def __init__(self, *layers):
        self.layers = layers
        self.out = None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return x
    
    def _backward(self, labels: np.ndarray):
        dY = self._squared_error(self.out, labels)
        for layer in reversed(self.layers):
            dY = layer._backward(dY)
        return 0
    
    def _squared_error(self, preds: np.ndarray, labels: np.ndarray) -> float:
        return preds - labels
    
    def _gradient_step(self, lr: float = 0.01):
        for layer in self.layers:
            if isinstance(layer, Layer) and layer.requires_grad:
                layer.weights -= lr * layer.dW
                layer.b -= lr * layer.db
                layer.dW = None
                layer.db = None
                layer.cache = None
    
    def train(self, x: np.ndarray, labels: np.ndarray, epochs: int = 10, batch_size: int = 32, lr: float = 0.01):
        n_samples = x.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(n_samples)
            x_perm = x[perm]
            labels_perm = labels[perm]
            for i in range(0, n_samples, batch_size):
                x_batch = x_perm[i:i+batch_size]
                labels_batch = labels_perm[i:i+batch_size]
                self(x_batch)
                self._backward(labels_batch)
                self._gradient_step(lr)
            if (epoch + 1) % 1 == 0 or epoch == 0:
                loss = np.mean((self.out - labels_batch) ** 2)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
