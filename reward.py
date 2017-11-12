
class Reward:
    def reward(self, board):
        pass


class GAReward(Reward):

    def __init__(self, aggregate_weight_w=-0.51, lines_w=0.76, holes_w=-0.35, bumpiness_w=-0.18):
        self._aggregate_weight_w = aggregate_weight_w
        self._lines_w = lines_w
        self._holes_w = holes_w
        self._bumpiness_w = bumpiness_w

    def reward(self, board):
        return self._aggregate_weight_w * self.aggregate_height(board) + \
               self._lines_w * self.lines(board) + \
               self._holes_w * self.holes(board) + \
               self._bumpiness_w * self.bumpiness(board)

    def column_height(self, board, col):
        rows = len(board)
        for row in range(rows):
            if board[row][col] != 0:
                return rows - row

        return 0

    def aggregate_height(self, board):
        """
        score 1
        그리드가 얼마나 높은지는 말한다. 이를 계산하기 위해, 각 컬럼의 높이를 합한다.
        :return:
        """
        return sum([self.column_height(board, c) for c in range(len(board[0]))])

    def holes(self, board):
        """
        score 2
        중간에 비어있는 공간을 말하며, 하나의 타일이 1이다.
        :return:
        """

        count = 0

        cols = len(board[0])
        rows = len(board)

        for c in range(cols):
            block = False
            for r in range(rows):
                if board[r][c] != 0:
                    block = True
                elif (board[r][c] == 0) & block:
                    count += 1

        return count

    def bumpiness(self, board):
        """
        score 3
        우물의 스코어를 구한다
        인접한 컬럼과의 차이합
        :return:
        """
        cols = len(board[0])

        return sum([abs(self.column_height(board, col) - self.column_height(board, col+1)) for col in range(cols-1)])

    def lines(self, board):
        """
        score 4
        그리드에서 clear될 라인의 수.
        :return:
        """

        rows = len(board)

        return len([row for row in range(rows - 1) if self.is_line(board, row)])

    def is_line(self, board, row):
        return 0 not in board[row]

if __name__ == '__main__':
    reward = GAReward()

    board = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        # [0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
        # [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    #
    # reward = Reward()
    # holes = reward.holes(board=board)
    #
    value = reward.lines(board)

    print(value)





