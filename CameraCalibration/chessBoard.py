"""
this file is used to define the chessboard
"""


class ChessBoard:
    def __init__(self, col, row, width) -> None:
        self.col = col  # num of corners' cols(the odd side)
        self.row = row  # num of corners' rows(the even side)
        self.width = width  # length(mm) between rows/cols in real world
