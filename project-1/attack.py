# import and build the model, train it, and attack it with L-BFGS-B

from src import Layer, Sequential, ReLU, Sigmoid
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
from scipy.optimize import minimize


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

    # train model on mnist
    np.random.seed(111)
    
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
    model.train(X_train, y_train, X_valid, y_valid, epochs=1, batch_size=32, lr=0.1)

    # evaluate on validation set (accuracy)
    model.get_accuracy(X_valid, y_valid, type="Validation")

# ---------------------------------------------------------------------------------------------------
    
    # define the attack


    # L-BFGS-B Attack
    def generate_new_image(x, target_label, model, c=0.001):
        y_target = np.zeros(10, dtype=int)
        y_target[target_label] = 1
        bounds = [(-xi, 1 - xi) for xi in x]

        def softmax(X):
            exps = np.exp(X)
            return exps / np.sum(exps)

        def cross_entropy(x_adv, y_onehot):
            logits = model(x_adv.reshape(1, -1))[0]
            p = softmax(logits)
            return -np.sum(y_onehot * np.log(p + 1e-8))

        res = minimize(
            lambda r: c * np.linalg.norm(r)**2 + cross_entropy(np.clip(x + r, 0, 1), y_target),
            np.zeros_like(x),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500}
        )

        return np.clip(x + res.x, 0, 1)
    
    def generate_additional_data(X, y, model, num_samples=500):
        indices = np.random.choice(len(X), num_samples, replace=False)
        X_adv = []
        y_adv = []

        for i in indices:
            x = X[i]
            true_label = np.argmax(y[i])
            target_label = (true_label + 1) % 10
            x_adv = generate_new_image(x, target_label, model)
            X_adv.append(x_adv)
            y_adv.append(y[i])  # keep original label

        return np.array(X_adv), np.array(y_adv)
    
# ---------------------------------------------------------------------------------------------------
    
    # perform the attack and train on the new data

    x = X_test[40]
    true_label = np.argmax(y_test[40])
    target_label = (true_label + 1) % 10

    x_adv = generate_new_image(x, target_label, model)

    pred_label = np.argmax(model(x_adv.reshape(1, -1)))
    print(f"True label: {true_label}, Target label: {target_label}, Predicted on adversarial: {pred_label}")

    X_adv, y_adv = generate_additional_data(X_train, y_train, model, num_samples=500)

    X_new = np.vstack([X_train, X_adv])
    y_new = np.vstack([y_train, y_adv])

    model.train(X_new, y_new, epochs=5, batch_size=32, lr=0.01)

    model.get_accuracy(X_new, y_new, type="Training")
