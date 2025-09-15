import numpy as np
from scipy.optimize import minimize

 # L-BFGS-B attack functions
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
