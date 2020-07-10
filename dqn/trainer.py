import tensorflow as tf
import gym
import fire
import gc
import os

from tqdm import tqdm

from typing import NamedTuple, List, Any, Optional

from .policy import DQNPolicy
from .memory import ReplayBuffer, DQNConverter
from .state import DQNState


class DQNTrainer:
    def __init__(self,
                 environment: str = 'VideoPinball-v0',
                 log_dir: str = './logs'):
        self.environment = environment
        self.log_dir = log_dir

    def _initialize(self, env, max_episode_steps, height, width,
                    agent_history_length, replay_buffer, size,
                    frame_skip_decimation, num_tracked_states):
        with tf.device('/CPU:0'):
            progress_bar = tqdm(total=max(size, num_tracked_states),
                                unit="sample")
            num_transitions = 0
            random_states = []
            while True:
                env.reset()
                initial_frame = env.render(mode='rgb_array')

                dqn_state = DQNState(height, width, agent_history_length)
                sequence = dqn_state.reset(initial_frame)

                action = None
                for t in range(int(max_episode_steps)):
                    if t % frame_skip_decimation == 0:
                        action = env.action_space.sample()

                        observation, reward, done, info = env.env.step(
                            int(action))
                        frame = env.render(mode='rgb_array')

                        next_sequence = dqn_state.update(frame)

                        reward = tf.clip_by_value(reward, -1., 1.)
                        if num_transitions < size:
                            transition = (sequence,
                                          tf.convert_to_tensor(action,
                                                               dtype=tf.int64),
                                          tf.identity(reward), next_sequence,
                                          tf.identity(done))

                            replay_buffer.append(transition)
                            num_transitions += 1

                        if not done and len(
                                random_states) < num_tracked_states:
                            random_states.append(sequence)

                        sequence = next_sequence

                        progress_bar.update()

                        if num_transitions == size and len(
                                random_states) == num_tracked_states:
                            return tf.stack(random_states, 0)

                        if done:
                            break
                    else:
                        observation, true_reward, done, info = env.step(action)
                        frame = env.render(mode='rgb_array')
                        sequence = dqn_state.update(frame)
                        if done:
                            break

    def run(self,
            discount_factor: float = 0.99,
            learning_rate: float = 2.5e-4,
            momentum: float = 0.95,
            min_steps: int = 100000,
            replay_buffer_capacity: int = int(1e5),
            max_episode_steps: int = int(1e6),
            seed: int = 5,
            batch_size: int = 32,
            frame_skip_decimation: int = 4,
            initial_epsilon: float = 1.,
            min_epsilon: float = 0.1,
            epsilon_schedule_duration: int = int(1e6),
            target_network_update_decimation: int = 10000,
            agent_history_length: int = 4,
            summary_decimation: int = 100,
            num_tracked_states: int = 32,
            replay_buffer_start_size: int = 50000,
            height: int = 110,
            width: int = 84,
            device_name: Optional[str] = None,
            clip_by_norm: Optional[float] = None):
        os.makedirs(self.log_dir, exist_ok=True)
        with tf.device(device_name):
            tf.random.set_seed(seed)
            env = gym.make(self.environment)
            try:
                assert isinstance(
                    env.action_space,
                    gym.spaces.Discrete), ('must be a discrete action space')

                num_actions = env.action_space.n
                policy = DQNPolicy(num_actions)
                target_policy = DQNPolicy(num_actions)

                optimizer = tf.keras.optimizers.RMSprop(
                    learning_rate=learning_rate, momentum=momentum)

                # Saved hyperparameters and state variables.
                discount_factor = tf.Variable(discount_factor,
                                              dtype=tf.float32,
                                              trainable=False)
                epsilon = tf.Variable(initial_epsilon,
                                      dtype=tf.float32,
                                      trainable=False)
                min_epsilon = tf.Variable(min_epsilon,
                                          dtype=tf.float32,
                                          trainable=False)
                initial_epsilon = tf.Variable(initial_epsilon,
                                              dtype=tf.float32,
                                              trainable=False)
                epsilon_schedule_duration = tf.Variable(
                    epsilon_schedule_duration, dtype=tf.int64, trainable=False)
                step = tf.Variable(0, dtype=tf.int64, trainable=False)
                episode = tf.Variable(0, dtype=tf.int64, trainable=False)
                frame_skip_decimation = tf.Variable(frame_skip_decimation,
                                                    dtype=tf.int32,
                                                    trainable=False)
                target_network_update_decimation = tf.Variable(
                    target_network_update_decimation,
                    dtype=tf.int64,
                    trainable=False)
                agent_history_length = tf.Variable(agent_history_length,
                                                   dtype=tf.int32,
                                                   trainable=False)
                replay_buffer_capacity = tf.Variable(
                    int(replay_buffer_capacity),
                    dtype=tf.int64,
                    trainable=False)
                tracked_states = tf.Variable(tf.zeros([
                    num_tracked_states, height, width,
                    int(agent_history_length)
                ],
                                                      dtype=tf.float32),
                                             dtype=tf.float32,
                                             trainable=False)

                checkpoint = tf.train.Checkpoint(
                    policy=policy,
                    target_policy=target_policy,
                    optimizer=optimizer,
                    discount_factor=discount_factor,
                    frame_skip_decimation=frame_skip_decimation,
                    target_network_update_decimation=
                    target_network_update_decimation,
                    agent_history_length=agent_history_length,
                    replay_buffer_capacity=replay_buffer_capacity,
                    epsilon=epsilon,
                    tracked_states=tracked_states,
                    step=step,
                    episode=episode)

                manager = tf.train.CheckpointManager(checkpoint, self.log_dir,
                                                     10)
                ckpt = manager.restore_or_initialize()
                if ckpt is not None:
                    print('Restored Latest Checkpoint')
                else:
                    print('Initialized Checkpoint')

                replay_buffer_path = os.path.join(self.log_dir,
                                                  'replay_buffer.raw')
                replay_buffer = ReplayBuffer(
                    replay_buffer_path, int(replay_buffer_capacity),
                    DQNConverter(height, width, int(agent_history_length)))

                if ckpt is None:
                    print('Initializing Replay Buffer')
                    replay_buffer.initialize()

                start_step = int(step)
                end_step = start_step + min_steps
                delta_epsilon = (min_epsilon - initial_epsilon
                                 ) / float(epsilon_schedule_duration)

                try:
                    replay_buffer.open()

                    if ckpt is None:
                        print('Initializing Trainer...')
                        tracked_states.assign(
                            self._initialize(env, max_episode_steps, height,
                                             width, int(agent_history_length),
                                             replay_buffer,
                                             replay_buffer_start_size,
                                             frame_skip_decimation,
                                             num_tracked_states))

                    summary_writer = tf.summary.create_file_writer(
                        self.log_dir)
                    huber_loss = tf.keras.losses.Huber()
                    while step < end_step:
                        env.reset()
                        initial_frame = env.render(mode='rgb_array')

                        dqn_state = DQNState(height, width,
                                             agent_history_length)
                        sequence = dqn_state.reset(initial_frame)

                        if step == 0:
                            print('Initializing Target Policy Weights')
                            # Build the models.
                            batched_sequence = tf.expand_dims(sequence, 0)
                            policy.dqn(batched_sequence)
                            target_policy.dqn(batched_sequence)

                            # Copy the policy weights to the target.
                            target_policy.copy_policy_weights(policy)

                        average_reward = tf.keras.metrics.Mean()
                        average_true_reward = tf.keras.metrics.Mean()
                        total_true_reward = 0
                        total_reward = 0
                        action = None
                        for t in range(int(max_episode_steps)):
                            if t % frame_skip_decimation == 0:
                                action = policy.epsilon_greedy_sample(
                                    sequence, epsilon)

                                observation, true_reward, done, info = env.step(
                                    int(action))
                                frame = env.render(mode='rgb_array')
                                next_sequence = dqn_state.update(frame)

                                reward = tf.clip_by_value(true_reward, -1., 1.)

                                total_true_reward += true_reward
                                total_reward += reward

                                with tf.device('/CPU:0'):
                                    transition = (tf.identity(sequence),
                                                  tf.identity(action),
                                                  tf.identity(reward),
                                                  tf.identity(next_sequence),
                                                  tf.identity(done))
                                    replay_buffer.append(transition)

                                (state, a, r, state_next, is_terminal
                                 ) = replay_buffer.sample_batch(batch_size)

                                q_values_next = target_policy.dqn(
                                    tf.stop_gradient(state_next))
                                q_values_next_max = tf.reduce_max(
                                    q_values_next, axis=-1)
                                with tf.GradientTape() as tape:
                                    q_values_current = policy.dqn(
                                        tf.stop_gradient(state))
                                    q_values_current_action = tf.gather(
                                        q_values_current,
                                        tf.stop_gradient(a),
                                        axis=-1,
                                        batch_dims=1)

                                    target = tf.where(
                                        tf.stop_gradient(is_terminal),
                                        tf.stop_gradient(r),
                                        tf.stop_gradient(r) +
                                        tf.stop_gradient(discount_factor) *
                                        q_values_next_max)

                                    mean_error = huber_loss(
                                        target, q_values_current_action)

                                grads = tape.gradient(
                                    mean_error, policy.dqn.trainable_variables)
                                if clip_by_norm is not None:
                                    processed_grads = [
                                        tf.clip_by_norm(g, clip_by_norm)
                                        for g in grads
                                    ]
                                else:
                                    processed_grads = grads

                                optimizer.apply_gradients(
                                    zip(processed_grads,
                                        policy.dqn.trainable_variables))

                                print('Episode', episode.numpy(), '| Step',
                                      step.numpy(), ': True Reward =',
                                      float(true_reward), ', Clipped Reward =',
                                      float(reward), ', Mean Error =',
                                      mean_error.numpy(), ', Epsilon =',
                                      float(epsilon), ', Action =',
                                      int(action))

                                average_true_reward.update_state(true_reward)
                                average_reward.update_state(reward)

                                if (t // frame_skip_decimation
                                    ) % summary_decimation == 0:
                                    tracked_states_q = policy.dqn(
                                        tracked_states)
                                    tracked_states_q_max = tf.reduce_max(
                                        tracked_states_q, axis=-1)
                                    tracked_states_average_q = tf.reduce_mean(
                                        tracked_states_q_max)
                                    with summary_writer.as_default():
                                        tf.summary.scalar('error',
                                                          mean_error,
                                                          step=step)
                                        tf.summary.scalar('epsilon',
                                                          float(epsilon),
                                                          step=step)
                                        tf.summary.scalar(
                                            'tracked_states_average_q',
                                            tracked_states_average_q,
                                            step=step)

                                sequence = next_sequence

                                step.assign_add(1)

                                epsilon.assign(
                                    tf.maximum(min_epsilon,
                                               epsilon + delta_epsilon))

                                if step % target_network_update_decimation == 0:
                                    print(
                                        'Copying Q-Network to Target Network')
                                    target_policy.copy_policy_weights(policy)

                                if done:
                                    break
                            else:
                                observation, true_reward, done, info = env.step(
                                    int(action))
                                frame = env.render(mode='rgb_array')
                                sequence = dqn_state.update(frame)
                                reward = tf.clip_by_value(true_reward, -1., 1.)
                                average_true_reward.update_state(reward)
                                average_reward.update_state(reward)
                                total_true_reward += true_reward
                                total_reward += reward
                                if done:
                                    break

                        episode_length = int(t + 1)
                        episode_return = average_reward.result()
                        episode_true_return = average_true_reward.result()

                        print('Episode', episode.numpy(), '|',
                              'Average True Reward =',
                              episode_true_return.numpy(),
                              ', Average Clipped Reward =',
                              episode_return.numpy(), ', Total True Reward =',
                              float(total_true_reward), ', Total Reward =',
                              float(total_reward), ', Episode Length =',
                              episode_length)

                        with summary_writer.as_default():
                            tf.summary.scalar('episode_number',
                                              int(episode),
                                              step=step)
                            tf.summary.scalar('episode_length',
                                              episode_length,
                                              step=step)
                            tf.summary.scalar('average_reward_per_episode',
                                              episode_return,
                                              step=step)
                            tf.summary.scalar('average_score_per_episode',
                                              episode_true_return,
                                              step=step)
                            tf.summary.scalar('total_score_per_episode',
                                              total_true_reward,
                                              step=step)
                            tf.summary.scalar('total_reward_per_episode',
                                              total_reward,
                                              step=step)

                        episode.assign_add(1)
                        manager.save()
                        gc.collect()

                    manager.save()
                finally:
                    replay_buffer.close()
            finally:
                env.close()


if __name__ == '__main__':
    fire.Fire(DQNTrainer)
