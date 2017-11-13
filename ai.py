import numpy as np
import random
import copy
from pygame_tetris import *
import time


class AI_1:

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

    def nomalize_board(self, board):
        return [[1 if col != 0 else 0 for col in row] for row in board]

    def step(self, state, actions):
        """
        행위를 진행한다
        :param action:
        :return:
        """

        states = []
        new_state = self.nomalize_board(state)
        max_reward = 0

        for action in actions:

            print(action)

            reward_value = self.game.key_actions[self.game.actions[action]]()

            if not reward_value:
                reward_value = 0

            if self.game.gameover:
                reward_value = -100

            if reward_value >= max_reward:
                max_reward = reward_value

            next_state = self.nomalize_board(self.game.reshape_board())

            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", next_state)

            if reward_value > 0:
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", reward_value)

            states.append((new_state, action, reward_value, next_state, self.game.gameover))

            new_state = next_state

            time.sleep(0.5)

        if max_reward > 0:
            new_states = []
            for state in states:
                temp = list(state)
                temp[2] = max_reward
                new_states.append(tuple(temp))

            return new_states
        else:
            return states

class AI_2:
    action_map = [
        [0],
        [0, 0],
        [0, 0, 0],
        [],
        [1],
        [1, 1],
        [1, 1, 1],
        [1, 1, 1, 1],

        [2, 0],
        [2, 0, 0],
        [2, 0, 0, 0],
        [2],
        [2, 1],
        [2, 1, 1],
        [2, 1, 1, 1],
        [2, 1, 1, 1, 1],

        [2, 2, 0],
        [2, 2, 0, 0],
        [2, 2, 0, 0, 0],
        [2, 2],
        [2, 2, 1],
        [2, 2, 1, 1],
        [2, 2, 1, 1, 1],
        [2, 2, 1, 1, 1, 1],

        [2, 2, 2, 0],
        [2, 2, 2, 0, 0],
        [2, 2, 2, 0, 0, 0],
        [2, 2, 2],
        [2, 2, 2, 1],
        [2, 2, 2, 1, 1],
        [2, 2, 2, 1, 1, 1],
        [2, 2, 2, 1, 1, 1, 1]
    ]

    def __init__(self, input_size, output_size, game):
        self.input_size = input_size
        self.output_size = output_size
        self.game = game

    def deep_copy_board(self, board):
        return copy.deepcopy(board)

    # def get_possible_actions(self):
    #     actions = []
    #     change = 0
    #
    #     for i in range(4):
    #         stone_y = self.game.stone_y
    #
    #         for stone_x in (range(-self.game.stone_x, self.game.cols - (self.game.stone_x + len(self.game.stone[0])))):
    #
    #             action = (stone_x, change)
    #
    #             if not check_collision(self.game.board, self.game.stone, (stone_x, stone_y)):
    #                 if action not in actions:
    #                     actions.append(action)
    #
    #             stone_y = self.game.stone_y
    #
    #         self.game.rotate_stone()
    #         change += 1
    #
    #     self.game.rotate_stone()
    #
    #     return actions

    def get_action_score(self, moves):

        rotate = 0
        new_x = 0

        for move in moves:
            if move == 2:
                rotate += 1
            elif move == 0:
                new_x -= 1
            elif move == 1:
                new_x += 1

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

    def get_best_action(self):

        best_score = None
        best_action = None

        for action in self.action_map:
            score = self.get_action_score(action)

            if not best_score:
                best_score = score
                best_action = action
            elif score >= best_score:
                best_score = score
                best_action = action

        return self.action_map.index(best_action)

    def sample(self):
        """
        0:LEFT
        1:RIGTH
        2:UP
        3:RETURN
        :return:
        """
        action = self.get_best_action()

        print(action)

        return [action]

    def nomalize_board(self, board):
        return [[1 if col != 0 else 0 for col in row] for row in board]

    def join_block_into_board(self, board):
        return join_matrixes(board, self.game.stone, (self.game.stone_x, self.game.stone_y))

    def step(self, state, action):
        """
        행위를 진행한다
        :param action:
        :return:
        """

        new_state = copy.deepcopy(self.game.board)
        new_state = self.join_block_into_board(new_state)
        new_state = self.nomalize_board(new_state)
        new_state = self.game.reshape_board2(new_state)

        states = []

        action = action[0]

        for move in self.action_map[action]:
            self.game.key_actions[self.game.actions[move]]()

        reward_value = self.game.key_actions[self.game.actions[3]]()
        if not reward_value:
            reward_value = 0

        if self.game.gameover:
            reward_value = -100

        print(">>>>>>>>", reward_value)

        next_state = self.nomalize_board(self.game.reshape_board())

        states.append((new_state, action, reward_value, next_state, self.game.gameover))

        return states
