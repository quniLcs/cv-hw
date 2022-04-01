import tensorflow as tf

if __name__ == "__main__":
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    assert x_test.shape == (10000, 28, 28)
    assert y_test.shape == (10000,)

    # Model loading
    # Accuracy
