# import NN library from src and train and test it on MNIST

from src import Layer, Sequential, ReLU, Sigmoid, LeakyReLU
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
from scipy.optimize import minimize

if __name__ == "__main__":

    # training on MNIST dataset (classification)

    # load mnist.pkl.gz
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    X_test, y_test = test_set

    # one-hot encode y
    y_train = np.eye(10)[y_train]
    y_valid = np.eye(10)[y_valid]
    y_test = np.eye(10)[y_test]

    # set seed for reproducibility
    np.random.seed(111)

    # build model just like PyTorch API: https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    # (ReLU or LeakyReLU instead of Sigmoid for hidden layers would perform better and is available, but sticking to Sigmoid as per instructions)

    model = Sequential(
        Layer(28*28, 128),
        Sigmoid(),
        Layer(128, 64),
        Sigmoid(),
        Layer(64, 10),
        Sigmoid()
    )

    """
    # Example model using LeakyReLU (epochs = 10, batch_size = 32, lr = 0.01 gives ~97% accuracy on test set)
    leaky_model = Sequential(
        Layer(28*28, 128),
        LeakyReLU(alpha=0.01),           
        Layer(128, 64),
        LeakyReLU(alpha=0.01),  
        Layer(64, 10),
        Sigmoid()
    )"""

    # train model
    model.train(X_train, y_train, X_valid, y_valid, epochs=10, batch_size=32, lr=0.1)

    # evaluate on validation set
    model.get_accuracy(X_valid, y_valid, type="Validation")

    # evaluate on test data (do this once you have decided on your hyperparameters)
    model.get_accuracy(X_test, y_test, type="Test")



# ---------------------------------------------------------------------------------------------------



    # an autoencoder model to compress images to 64 dimensions and then reconstruct them (to see if the model can handle it [which it does] - a good test for the code)
    
    """model = Sequential(
        Layer(28*28, 256),
        ReLU(),
        Layer(256, 128),
        ReLU(),
        Layer(128, 64),
        ReLU(),
        Layer(64, 128),
        ReLU(),
        Layer(128, 256),
        ReLU(),
        Layer(256, 28*28),
        Sigmoid()
    )
    model.train(X_train, X_train, epochs=20, batch_size=64, lr=0.1)

    preds = model(X_valid)
    loss = np.mean((preds - X_valid) ** 2)
    print(f"Validation Loss: {loss:.4f}")
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_valid[i].reshape(28, 28), cmap='gray')
        plt.title("original")
        plt.axis('off')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(preds[i].reshape(28, 28), cmap='gray')
        plt.title("reconstructed")
        plt.axis('off')

    plt.show()"""

    

