# import model from src and train and test it here

from src import Layer, Sequential, ReLU, Sigmoid
import numpy as np

if __name__ == "__main__":
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    X = np.tile(X, (50, 1))
    y = np.tile(y, (50, 1))
    
    X += np.random.randn(*X.shape) * 0.1
    
    model = Sequential(
        Layer(2, 8),
        ReLU(),
        Layer(8, 4),
        ReLU(),
        Layer(4, 1),
        Sigmoid()
    )
    
    train_model = model.train(X, y, epochs=100, batch_size=16, lr=0.1)
    

