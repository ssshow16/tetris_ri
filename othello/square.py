class Square(object):
    """
    Represents one square of an othello board.

    Instance variables:
        value - If a disk is placed, its value, else None
        is_valid_move - boolean if the current player can play here
        flipped_disks - if the current player plays here,
                        which disks will flip to their color
    """

    def __init__(self, value=None):
        """
        Initialize Square.

        Optional arguments:
            value - current value of square, default None
        """
        self._value = value
        self.is_valid_move = False
        self.flipped_disks = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.is_valid_move = False
        self.flipped_disks = []