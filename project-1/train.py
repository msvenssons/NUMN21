# import model from src and train and test it here

from src import Layer, Sequential, ReLU, Sigmoid
import numpy as np

if __name__ == "__main__":
    print("Testing the model...")
    #np.random.seed(42)
    model = Sequential(Layer(5, 3), 
                       ReLU(),
                       Layer(3, 1),
                       Sigmoid())
    
    x = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [4, 4, 4, 4, 4]])
    labels = np.array([[1], [1], [1], [0]])
    out = model(x)
    model.train(x, labels, epochs=500, batch_size=1, lr=0.001)

    print(model(x))

