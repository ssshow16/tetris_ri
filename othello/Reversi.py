TILES = ['X', 'O']

class Reversi():
    def __init__(self, boardSize = 8):
        self.boardSize = 8
        self.tileIdx = 0
        self.tile = TILES[self.tileIdx]
        self.scores = [2, 2]
        
    def createNewBoard(self):
        self.board = [[' ' for x in range(self.boardSize)] for x in range(self.boardSize)]
        
    def draw(self):
        HLINE = '  +---+---+---+---+---+---+---+---+'
        VLINE = '  |   |   |   |   |   |   |   |   |'
    
        print( '    1   2   3   4   5   6   7   8')
        print(HLINE)

        for y in range(8):
            print (VLINE)
            print (y+1, end=' ')
            for x in range(8):
                print('| %s'%(self.board[x][y]), end=' ')
            print('|')
            print(VLINE)
            print(HLINE)
    
    def resetBoard(self):
        self.board = [[' ' for x in range(self.boardSize)] for x in range(self.boardSize)]
        half = int(self.boardSize / 2)
        self.board[half - 1][half - 1] = 'X'
        self.board[half - 1][half] = 'O'
        self.board[half][half - 1] = 'O'
        self.board[half][half] = 'X'
        
    def getOtherTile(self):
        return TILES[(self.tileIdx+1)%2]
    
    def isValidMove(self, xstart, ystart):
        if self.board[xstart][ystart] != ' ' or not self.isOnBoard(xstart, ystart):
            return False
        
        otherTile = self.getOtherTile()
        
        tilesToFlip = []
        for xDirection, yDirection in [[0,1], [1,1], [1, 0], [1,-1], [0,-1],[-1,-1],[-1,1], [-1, 0]]:
            x, y = xstart + xDirection, ystart + yDirection
            while self.isOnBoard(x, y) and self.board[x][y] == otherTile:
                x += xDirection
                y += yDirection
            if self.board[x][y] == self.tile:
                while True:
                    x -= xDirection
                    y -= yDirection
                    if x == xstart and y == ystart:
                        break
                    tilesToFlip.append([x,y])
        
        if len(tilesToFlip) == 0:
            return False
        return tilesToFlip

    
    def isOnBoard(self, x, y):
        return x >= 0 and x < self.boardSize and y >= 0 and y < self.boardSize

    def getBoardWithValidMoves(self, tile):
        dupeBoard = self.getBoardCopy()
        
        for x, y in self.getValidMoves(dupeBoard, tile):
            dupeBoard[x][y] = '.'
        return dupeBoard

    def getValidMoves(self):
        validMoves = []
        
        for x in range(8):
            for y in range(8):
                if self.isValidMove(x, y) != False:
                    validMoves.append([x, y])
        return validMoves

    def getBoardCopy(self):
        dupeBoard = [[' ' for x in range(self.boardSize)] for x in range(self.boardSize)]
        for x in range(8):
            for y in range(8):
                dupeBoard[x][y] = self.board[x][y]
        return dupeBoard

    def flipBoard(self, tilesToFlip):
        for x, y in tilesToFlip:
            self.board[x][y] = self.tile
        self.scores[self.tileIdx] += len(tilesToFlip)
        self.changePlayer()
        self.scores[self.tileIdx] -= len(tilesToFlip)-1
            
    def score(self):
        return {'X':self.scores[0], 'O':self.scores[1]}

    def makeMove(self, xstart, ystart):
        tilesToFlip = self.isValidMove(xstart, ystart)
        
        if tilesToFlip == False:
            print("Invalid move", [xstart, ystart])
            return False
        
        tilesToFlip.append([xstart, ystart])
        self.flipBoard(tilesToFlip)
        return True

    def changePlayer(self):
        self.tileIdx = (self.tileIdx + 1) % 2
        self.tile = TILES[self.tileIdx]
    
    def getState(self):
        return self.board, self.scores, self.tile


