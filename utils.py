def check_win(board):
    # Check for win in rows
    for row in board:
        if row[0] == row[1] == row[2] and row[0]:
            return row[0]

    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col]:
            return board[0][col]

    # Check for win in diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0]:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2]:
        return board[0][2]

    return False

def check_draw(board):
    for row in board:
        for cell in row:
            if not cell:
                return False
            
    return True