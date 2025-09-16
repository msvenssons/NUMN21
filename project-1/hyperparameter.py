# import model from src and train and test it here

from src import Layer, Sequential, ReLU, Sigmoid
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import itertools
import time


# Hyperparameter search: neural net size/depth

def _build_model_factory(input_dim, num_neurons, k_extra_layers, n_classes=10):
    """Return a function that builds the model when called.
    This avoids retaining references between runs if your framework does that."""
    def build():
        # Build Sequential model per your specified API.
        layers = []
        layers.append(Layer(input_dim, num_neurons))
        layers.append(ReLU())
        # add K extra hidden layers of size N (each followed by ReLU)
        for _ in range(k_extra_layers):
            layers.append(Layer(num_neurons, num_neurons))
            layers.append(ReLU())
        # output
        layers.append(Layer(num_neurons, n_classes))
        layers.append(Sigmoid())
        return Sequential(*layers)
    return build

def test_architectures(X_train, y_train, X_valid, y_valid,
                       N_values=None, K_values=None,
                       batch_size=64, lr=1.0, epochs=20,
                        n_classes=10):
   
    # defaults
    if N_values is None:
        N_values = list(range(10, 101, 10))   # 10,20,...,100
    if K_values is None:
        K_values = list(range(0, 6))         # K = 0..5
    
    input_dim = X_train.shape[1]
    
    rows = list()
    # iterate
    for N in N_values:
        for K in K_values:
            print(f"{N},{K}")
            num_hidden_layers = 1 + K
            # compute total parameters
            layer_sizes = [input_dim] + [N] * K + [N]  # input + hidden layers
            total_params = 0
            for i in range(len(layer_sizes)-1):
                total_params += layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]  # weights + biases
            # add output layer
            total_params += N * n_classes + n_classes
            
            # build model
            build_model = _build_model_factory(input_dim, N, K, n_classes=n_classes)
            model = build_model()
            
            # measure training time
            t0 = time.time()
            model.train(X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr)
            elapsed = time.time()-t0
            
            # compute train acc
            out_train = model(X_train)
            preds_train = np.argmax(out_train, axis=1)
            labels_train = np.argmax(y_train, axis=1)
            train_acc = np.mean(preds_train == labels_train)
            
            # compute valid acc
            out_valid = model(X_valid)
            preds_valid = np.argmax(out_valid, axis=1)
            labels_valid = np.argmax(y_valid, axis=1)
            val_acc = np.mean(preds_valid == labels_valid)
            
            # print CSV row
            row = f"{total_params},{num_hidden_layers},{N},{K},{epochs},{batch_size},{lr},{train_acc:.6f},{val_acc:.6f},{elapsed:.3f}"
            rows.append(row)
    print("total_params,num_hidden_layers,neurons_per_layer,K,epochs,batch_size,lr,train_acc,val_acc,time_seconds")
    print("\n".join(rows))


# Compare one-hot and bit encoding.
def bit_vs_onehot():

    epochs=100
    batch_size=64
    lr=1

    model = Sequential(
        Layer(28*28, 30),
        ReLU(),
        Layer(30, 10),
        Sigmoid()
        )

    model.train(X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr)

    preds = model(X_valid)
    preds = np.argmax(preds, axis=1)
    labels = np.argmax(y_valid, axis=1)
    accuracy = np.mean(preds == labels)
    
    preds2 = model(X_train)
    preds2 = np.argmax(preds2, axis=1)
    labels2 = np.argmax(y_train, axis=1)
    accuracy2 = np.mean(preds2 == labels2)

    print(f"One-hot: lr={lr}, batch_size={batch_size}, epochs={epochs}, training accuracy={accuracy2}, validation accuracy={accuracy}")

    model = Sequential(
        Layer(28*28, 30),
        ReLU(),
        Layer(30, 4),
        Sigmoid()
        )

    model.train(X_train, y_train_b, epochs=epochs, batch_size=batch_size, lr=lr)


    preds = model(X_valid)
    preds = np.round(preds)
    labels = np.round(y_valid_b)
    accuracy = np.mean((preds == labels).all(axis=1))

    preds2 = model(X_train)
    preds2 = np.round(preds2)
    labels2 = np.round(y_train_b)
    accuracy2 = np.mean((preds2 == labels2).all(axis=1))

    print(f"Bit: lr={lr}, batch_size={batch_size}, epochs={epochs}, training accuracy={accuracy2}, validation accuracy={accuracy}")


# Hyperparameter search: lr+batch size 
def test_lr_batch():
    lrs = [0.5, 0.1, 0.05, 0.01] #0.5, 0.1, 0.05, 0.01, 0.005, 0.001
    batch_sizes = [128,64,32,16,8,4] #256,128,64,32,16,8,4
    epochs = [50]

    results = list()

    model1 = Sequential(
        Layer(28*28, 30),
        ReLU(),
        Layer(30, 10),
        Sigmoid()
        )

    # Pre-train with high lr for a while(same for all) for faster computation, so lower lr's can be compared. (can be commented out if not wanted)
    model1.train(X_train, y_train, epochs=20, batch_size=64, lr=1)

    for lr,batch_size,epoch in itertools.product(lrs,batch_sizes,epochs):

        model = deepcopy(model1)
        
        start = time.time()
        model.train(X_train, y_train, epochs=epoch, batch_size=batch_size, lr=lr)
        end = time.time()

        preds = model(X_valid)
        preds = np.argmax(preds, axis=1)
        labels = np.argmax(y_valid, axis=1)
        accuracy = np.mean(preds == labels)

        preds2 = model(X_train)
        preds2 = np.argmax(preds2, axis=1)
        labels2 = np.argmax(y_train, axis=1)
        accuracy2 = np.mean(preds2 == labels2)
 
        seconds = end-start

        #result = f"lr={lr}, batch_size={batch_size}, epochs={epoch}, training accuracy={accuracy2}, validation accuracy={accuracy}, time taken={seconds}"
        result = f"{lr},{batch_size},{epoch},{accuracy2},{accuracy},{seconds}"
        print(result)
        results.append(result)

    print("\n \n")
    print("lr,batch_size,epochs,training accuracy,validation accuracy,time taken")
    for result in results:
        print(result)


if __name__ == "__main__":
    
    np.random.seed(111)

    # training on MNIST dataset (classification)

    # load mnist.pkl.gz
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    X_test, y_test = test_set

    # bit encode y

    y_train_b = ((y_train[:, None] & (1 << np.arange(4)[::-1])) > 0).astype(int)
    y_valid_b = ((y_valid[:, None] & (1 << np.arange(4)[::-1])) > 0).astype(int)
    y_test_b = ((y_test[:, None] & (1 << np.arange(4)[::-1])) > 0).astype(int)

    # one-hot encode y
    y_train = np.eye(10)[y_train]
    y_valid = np.eye(10)[y_valid]
    y_test = np.eye(10)[y_test]

    
    #bit_vs_onehot()

    #test_lr_batch()

    test_architectures(X_train, y_train, X_valid, y_valid,N_values=[80],K_values=[1],batch_size=64, lr=0.1, epochs=3)



"""
Results:

Single hidden layer, varying neurons(lr=1,batch=64,epochs=50):

total_params,num_hidden_layers,neurons_per_layer,K,epochs,batch_size,lr,train_acc,val_acc,time_seconds
3985,1,5,0,50,64,1,0.270320,0.267000,49.090
7960,1,10,0,50,64,1,0.835420,0.829200,52.625
15910,1,20,0,50,64,1,0.963380,0.952900,65.128
31810,1,40,0,50,64,1,0.986580,0.970600,129.712
63610,1,80,0,50,64,1,0.994620,0.977200,193.065
127210,1,160,0,50,64,1,0.996220,0.981500,331.489
254410,1,320,0,50,64,1,0.996920,0.983000,350.645


One more hidden layer:

70090,2,80,1,50,64,1,0.996920,0.979100,139.571


More layers and/or using CE -> needs to lower lr


"""




