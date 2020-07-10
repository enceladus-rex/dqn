import tensorflow as tf

from .models import DQN


class DQNPolicy(tf.Module):
    def __init__(self, num_actions: int):
        self.dqn = DQN(num_actions)
        self.num_actions = num_actions

    def get_weights(self):
        return self.dqn.get_weights()

    def set_weights(self, weights):
        return self.dqn.set_weights(weights)

    def copy_policy_weights(self, policy: 'DQNPolicy'):
        return self.set_weights(policy.get_weights())

    def state_action_value(self, state, action):
        return self.dqn(x)[action]

    def epsilon_greedy_sample(self, state, epsilon: float):
        if tf.random.uniform((), 0, 1) <= epsilon:
            return self.random_action()
        else:
            return self.greedy_action(state)

    def random_action(self):
        return tf.random.uniform((), 0, self.num_actions, dtype=tf.int64)

    def greedy_action(self, state):
        q_values = tf.squeeze(self.dqn(tf.expand_dims(state, 0)), 0)
        return tf.argmax(q_values)
