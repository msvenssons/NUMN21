# import model from src and train and test it here

from src import Layer, Sequential, ReLU, Sigmoid
import numpy as np
import pickle
import gzip

# bug in model: check that output dim matches input dim (numpy adapts apparently but we would want an error)

if __name__ == "__main__":

    # training on MNIST dataset

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

    # define model (using ReLU instead of Sigmoid for hidden layers as this performs better)
    # precis som PyTorch:
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

    

