"""Implement Conway's Game of Life."""

import numpy as np
from matplotlib import pyplot
from scipy.signal import convolve2d

glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])

blinker = np.array([[0, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0]])

glider_gun = np.array([
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0]
])


class Game:
    """Have methods to play Conway's game of life."""

    def __init__(self, size):
        self.board = np.zeros((size, size))

    def play(self):
        """Run to play game of life."""
        print("Playing life. Press ctrl + c to stop.")
        pyplot.ion()
        while True:
            self._move()
            self._show()
            pyplot.pause(0.0000005)

    def _move(self):
        """Move the game one step forward by the required rules."""
        stencil = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbour_count = convolve2d(self.board, stencil, mode='same')

        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                self.board[i, j] = 1 if (neighbour_count[i, j] == 3 or
                                         (neighbour_count[i, j] == 2 and
                                          self.board[i, j])) else 0

    def __setitem__(self, key, value):
        """Let user set value of self.board by using []."""
        self.board[key] = value

    def _show(self):
        """Show the plot of the game."""
        pyplot.clf()
        pyplot.matshow(self.board, fignum=0, cmap='binary')
        pyplot.show()

    def insert(self, P, a):
        """Insert pattern P at location a."""
        x = P.grid.shape[0]
        y = P.grid.shape[1]
        x1 = a[0]
        y1 = a[1]
        for i in range(x1 - x//2, x1 + x//2 + 1):
            for j in range(y1 - y//2, y1 + y//2 + 1):
                self.board[i, j] = P.grid[i - (x1 - x//2), j - (y1 - y//2)]
        return self


class Pattern:
    """Have methods to return transformed pattern."""

    def __init__(self, arr):
        self.grid = arr

    def flip_vertical(self):
        """Return a new pattern which is upside down."""
        return Pattern(self.grid[::-1])

    def flip_horizontal(self):
        """Return a new pattern which is reversed left-right."""
        return Pattern(self.grid[:, ::-1])

    def flip_diag(self):
        """Return a new pattern which is the transpose of the original."""
        return Pattern(np.transpose(self.grid))

    def rotate(self, n):
        """Return rotation of n * 90-degree anti-clockwise of the original."""
        if n == 1:
            return Pattern(np.transpose(self.grid)[::-1])
        else:
            return Pattern.rotate(Pattern(np.transpose(self.grid)[::-1]), n-1)
