import numpy as np
import pytest
from src import Layer, Sequential, ReLU, LeakyReLU, Sigmoid


# Unit Tests (Testing the Individual Parts)

# Testing the activation functions

def test_relu_function():
    """Tests that the ReLU function correctly handles positive, negative, and zero values."""
    relu_layer = ReLU()
    test_input = np.array([-5, -1, 0, 3, 5])
    expected_output = np.array([0, 0, 0, 3, 5])

    # Run the function
    actual_output = relu_layer(test_input)

    # Check if the result is correct
    np.testing.assert_array_equal(actual_output, expected_output)

def test_sigmoid_function():
    """Tests that the Sigmoid function correctly handles values between 0 and 1."""
    sigmoid_layer = Sigmoid()
    test_input = np.array([0]) # Sigmoid of 0 should be 0.5
    expected_output = 0.5
    actual_output = sigmoid_layer(test_input)
    assert np.isclose(actual_output, expected_output)

def test_leaky_relu_function():
    """Tests that the LeakyReLU function correctly applies the alpha slope."""
    leaky_relu_layer = LeakyReLU(alpha=0.01)
    test_input = np.array([-100, -10, 0, 5, 10])
    expected_output = np.array([-1, -0.1, 0, 5, 10]) # -100*0.01 = -1, -10*0.01 = -0.1
    actual_output = leaky_relu_layer(test_input)
    assert np.all(np.isclose(actual_output, expected_output))


# Testing the Layer Class

def test_layer_initialization_shapes():
    """Tests if a Layer is created with the correct weight and bias shapes."""
    input_size = 784           # Each input has 784 features (like a 28x28 MNIST image)
    output_size = 30           # The layer should output vectors of size 30 (arbitrary for test)

    layer = Layer(input_size, output_size)  # Create a layer with the given input/output size

    # Check that the weights matrix has the correct shape: (input_features, output_features)
    assert layer.weights.shape == (input_size, output_size)

    # Check that the bias vector has the correct shape: (1 row, output_features)
    assert layer.b.shape == (1, output_size)

