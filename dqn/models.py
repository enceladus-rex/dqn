import tensorflow as tf


def preprocess_sequence(x: tf.Tensor, height: int, width: int) -> tf.Tensor:
    y = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.image.rgb_to_grayscale(y)

    original_shape = tf.shape(y)
    num_elements = original_shape[0] * original_shape[1]
    new_shape = tf.TensorShape(
        tf.concat([[num_elements], original_shape[2:]], axis=0))

    y_reshaped = tf.reshape(y, new_shape)
    y_reshaped = tf.image.resize(y_reshaped, [height, width])
    y_reshaped = tf.squeeze(y_reshaped, axis=-1)

    y = tf.reshape(y_reshaped,
                   [original_shape[0], original_shape[1], height, width])
    y = tf.transpose(y, [0, 2, 3, 1])

    return y


class DQN(tf.keras.Model):
    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.sequential_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 8, 4, activation='elu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, 4, 2, activation='elu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Reshape([-1]),
            tf.keras.layers.Dense(256, activation='elu'),
            tf.keras.layers.Dense(num_actions),
        ])

    def call(self, preprocessed: tf.Tensor) -> tf.Tensor:
        return self.sequential_layers(preprocessed)
