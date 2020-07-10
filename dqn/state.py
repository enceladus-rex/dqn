import tensorflow as tf

from .models import preprocess_sequence


class DQNState:
    def __init__(self, height: int, width: int, agent_history_length: int):
        self.height = height
        self.width = width
        self.agent_history_length = int(agent_history_length)
        self.state = None

    def _preprocess_frame(self, frame):
        expanded_frame = tf.expand_dims(tf.expand_dims(frame, 0), 0)
        preprocessed_frame = tf.squeeze(
            preprocess_sequence(expanded_frame, self.height, self.width), 0)
        return preprocessed_frame

    def reset(self, frame):
        preprocessed_frame = self._preprocess_frame(frame)
        self.state = tf.concat([preprocessed_frame] +
                               [tf.zeros_like(preprocessed_frame)] *
                               (self.agent_history_length - 1), -1)
        return self.state

    def update(self, frame):
        preprocessed_frame = self._preprocess_frame(frame)
        next_state = tf.concat([preprocessed_frame, self.state], -1)
        start, end = 0, min(next_state.shape[-1], self.agent_history_length)
        self.state = next_state[:, :, start:end]
        return self.state
