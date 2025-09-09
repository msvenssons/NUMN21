# import model from src and train and test it here

from src import Layer, Sequential, ReLU, Sigmoid
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt

# bug in model: check that output dim matches input dim (numpy adapts apparently but we would want an error)

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

    # train model on mnist
    np.random.seed(111)

    # define model (using ReLU instead of Sigmoid for hidden layers as it performs better)
    # precis som PyTorch: https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    model = Sequential(
        Layer(28*28, 128),
        ReLU(),
        Layer(128, 64),
        ReLU(),
        Layer(64, 10),
        Sigmoid()
    )

    # train model
    model.train(X_train, y_train, epochs=5, batch_size=32, lr=0.01)

    # evaluate on validation set
    preds = model(X_valid)
    preds = np.argmax(preds, axis=1)
    labels = np.argmax(y_valid, axis=1)
    #print(preds[0], labels[0])
    accuracy = np.mean(preds == labels)
    # use simple accuracy as a metric
    print(f"Validation Accuracy: {accuracy:.4f}")


    # eventually test on test data once we have narrowed down the hyperparameters



# ---------------------------------------------------------------------------------------------------

    # en autoencoder modell för att komprimera bilder till 64 dimensioner och sedan rekonstruera dem (för att se om modellen klarar det [det gör den] - ett bra test för koden)
    
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

    

