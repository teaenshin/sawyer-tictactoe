def check_win(board):
    # Check for win in rows
    for i in range(3):
        if board[3*i] == board[3*i + 1] == board[3*i + 2] and board[3*i]:
            return board[3*i]

    for i in range(3):
        if board[i] == board[3+i] == board[6+i] and board[i]:
            return board[i]


    # Check for win in diagonals
    if board[0] == board[4] == board[8] and board[0]:
        return board[0]
    if board[2] == board[4] == board[6] and board[2]:
        return board[2]

    return False

def check_draw(board):
    for spot in board:
        if not spot:
            return False

    return True