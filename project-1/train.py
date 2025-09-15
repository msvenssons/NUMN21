# import model from src and train and test it here

from src import Layer, Sequential, ReLU, Sigmoid
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
    # just like PyTorch: https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    model = Sequential(
        Layer(28*28, 128),
        Sigmoid(),
        Layer(128, 64),
        Sigmoid(),
        Layer(64, 10),
        Sigmoid()
    )

    # train model
    model.train(X_train, y_train, X_valid, y_valid, epochs=10, batch_size=32, lr=0.1)

    # evaluate on validation set
    def evaluate_model(model, X, y):
            preds = model(X)
            preds = np.argmax(preds, axis=1)
            labels = np.argmax(y, axis=1)
             #print(preds[0], labels[0])
            accuracy = np.mean(preds == labels)
             # use simple accuracy as a metric
            print(f"Validation Accuracy: {accuracy:.4f}")
    
    evaluate_model(model, X_valid, y_valid)


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

    evaluate_model(model, X_new, y_new)


    # eventually test on test data once we have narrowed down the hyperparameters



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

    

