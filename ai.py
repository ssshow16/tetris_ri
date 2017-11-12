import numpy as np
import random
import copy
from pygame_tetris import *
import time

class AI:

    def __init__(self, input_size, output_size, game):
        self.input_size = input_size
        self.output_size = output_size
        self.game = game

    def deep_copy_board(self, board):
        return copy.deepcopy(board)

    def get_possible_actions(self):
        actions = []
        change = 0

        for i in range(4):
            stone_y = self.game.stone_y

            for stone_x in (range(-self.game.stone_x, self.game.cols - (self.game.stone_x + len(self.game.stone[0])))):

                action = (stone_x, change)

                if not check_collision(self.game.board, self.game.stone, (stone_x, stone_y)):
                    if action not in actions:
                        actions.append(action)

                stone_y = self.game.stone_y

            self.game.rotate_stone()
            change += 1

        self.game.rotate_stone()

        return actions

    def get_action_score(self, action):

        rotate = action[1]
        new_x = action[0]

        new_board = copy.deepcopy(self.game.board)

        stone_x = self.game.stone_x
        stone_y = self.game.stone_y

        new_stone = self.game.stone

        for i in range(rotate):
            new_stone = rotate_clockwise(new_stone)

        # stone, stone_x, move
        stone_x = move_x(new_stone, stone_x, new_x)

        while not check_collision(new_board, new_stone, (stone_x, stone_y)):
            stone_y += 1

        stone_y += 1

        new_board = join_matrixes(new_board, new_stone, (stone_x, stone_y))

        reward_value = self.game.reward.reward(new_board)
        return reward_value

    def get_best_action(self, possible_actions):
        best_action = possible_actions[0]
        best_score = self.get_action_score(possible_actions[0])

        for action in possible_actions[1:]:
            score = self.get_action_score(action)

            if score >= best_score:
                best_score = score
                best_action = action

        actions = []

        for i in range(best_action[1]):
            actions.append(2)  # UP

        for i in range(abs(best_action[0])):
            if best_action[0] < 0:
                actions.append(0)  # LEFT
            else:
                actions.append(1)  # RIGHT

        actions.append(3)  # RETURN

        return actions

    def sample(self):
        """
        0:LEFT
        1:RIGTH
        2:UP
        3:RETURN
        :return:
        """
        action = np.zeros([self.output_size])
        action[random.randrange(self.output_size)] = 1

        possible_actions = self.get_possible_actions()
        actions = self.get_best_action(possible_actions)

        return actions

    def step(self, state, actions):
        """
        행위를 진행한다
        :param action:
        :return:
        """

        states = []
        new_state = state

        for action in actions:

            print(action)

            reward_value = self.game.key_actions[self.game.actions[action]]()

            if not reward_value:
                reward_value = 0

            if self.game.gameover:
                reward_value = -100

            next_state = self.game.reshape_board()

            if reward_value > 0:
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", reward_value)

            states.append((new_state, action, reward_value, next_state, self.game.gameover))

            new_state = next_state

            time.sleep(0.5)

            # print(new_state)

        return states
