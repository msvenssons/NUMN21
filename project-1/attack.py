# import and build the model, train it, and attack it with L-BFGS-B

from src import Layer, Sequential, ReLU, Sigmoid
import numpy as np
import pickle
import gzip
from utils import generate_new_image, generate_additional_data


if __name__ == "__main__":
# ---------------------------------------------------------------------------------------------------

    # data pre-processing and building model

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
    
    # build model
    model = Sequential(
        Layer(28*28, 128),
        Sigmoid(),
        Layer(128, 64),
        Sigmoid(),
        Layer(64, 10),
        Sigmoid()
    )

# ---------------------------------------------------------------------------------------------------

    # train model
    model.train(X_train, y_train, X_valid, y_valid, epochs=10, batch_size=32, lr=0.1)

    # evaluate on validation set (accuracy)
    model.get_accuracy(X_valid, y_valid, type="Validation")

# ---------------------------------------------------------------------------------------------------
    
    # perform the attack and train on the new data (additional functions are found in utils.py)

    def attack(model):
        x = X_test[40]
        true_label = np.argmax(y_test[40])
        target_label = (true_label + 1) % 10

        x_adv = generate_new_image(x, target_label, model)

        pred_label = np.argmax(model(x_adv.reshape(1, -1)))
        print(f"True label: {true_label}, Target label: {target_label}, Predicted on adversarial: {pred_label}")


    attack(model)
    X_adv, y_adv = generate_additional_data(X_test, y_test, model, num_samples=10)

    X_new = np.vstack([X_train, X_adv])
    y_new = np.vstack([y_train, y_adv])

    model.train(X_new, y_new, epochs=10, batch_size=32, lr=0.1)
    
    model.get_accuracy(X_valid, y_valid, type="Training")

    attack(model)
