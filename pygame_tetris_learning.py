import random
import threading
import time
from collections import deque

import tensorflow as tf
from dqn import DQN
from tetris.pygame_tetris import *

input_size = 10 * 23  # 4 ??
output_size = 4  # 4 LEFT, RIGHT, UP, RETURN

dis = 0.9
REPLAY_MEMORY = 50000

learning_rate = 1e-1


def simple_replay_train(DQN, train_batch):
    # 초기화 stack을 생성
    x_stack = np.empty(0).reshape(0, DQN.input_size)
    y_stack = np.empty(0).reshape(0, DQN.output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = DQN.predict(state)

        if done:
            Q[0, action] = reward
        else:
            # 네트워크를 통해 새로운 상태에 대한 Q 값을 갱신한다
            Q[0, action] = reward + dis * np.max(DQN.predict(next_state))

        # 학습을 하지 않고 쌓아두기만 한다.
        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    # 쌓아둔 값으로 학습을 한다
    return DQN.update(x_stack, y_stack)


def bot_play(mainDQN):
    '''
    학습된 네트워크를 가지고 시뮬레이션 해본다
    :param mainDQN:
    :return:
    '''
    env = TetrisApp()
    env.run()

    s = env.reset()  # 초기 상태값을 가져온다

    reward_sum = 0

    while True:
        # env.render()
        action = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(action)
        reward_sum += reward

        if done:
            print("Total score: {}".format(reward_sum))
            break


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    '''
    네트워크 간의 변수를 복사한다
    :param dest_scope_name:
    :param src_scope_name:
    :return:
    '''
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)

    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append((dest_var.assign(src_var.value())))

    return op_holder


def replay_train(mainDQN, targetDQN, train_batch):
    # 초기화 stack을 생성
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        if done:
            Q[0, action] = reward
        else:
            # 네트워크를 통해 새로운 상태에 대한 Q 값을 갱신한다
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        # 학습을 하지 않고 쌓아두기만 한다.
        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    # 쌓아둔 값으로 학습을 한다
    return mainDQN.update(x_stack, y_stack)


class Env(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.env = TetrisApp(Reward())

    def run(self):
        self.env.run()

    def sample(self):
        return self.env.sample()

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()


def main():
    num_episodes = 2000

    replay_buffer = deque()

    env = Env()
    env.start()

    with tf.Session() as sess:

        mainDQN = DQN(sess, input_size, output_size, name="main")
        targetDQN = DQN(sess, input_size, output_size, name="target")

        tf.global_variables_initializer().run()
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")

        for episode in range(num_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            state = env.reset()
            step_count = 0

            while not done:
                if np.random.rand(1) < e:
                    action = env.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, _ = env.step(action)

                # print(reward, done)z

                if done:
                    reward = -10000

                # 학습은 하지 않고 상태를 저장만 한다.
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:
                    break

                time.sleep(0.01)

            print("Episode:{} steps:{} ".format(episode, step_count))
            if step_count > 10000:
                pass

            if episode % 10 == 1:
                for _ in range(50):
                    # 랜덤하게 일부만 저장한다
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                print("Loss:", loss)
                sess.run(copy_ops)

        bot_play(mainDQN)


if __name__ == "__main__":
    main()









