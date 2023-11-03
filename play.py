import numpy as np
import cv2

# Robot (player 'robot') is always 'X'
# Human (player 'human') is always 'O'
# This should run independently of robot. The gamestate should accurately reflect any 2 players (robot or human) playing tictactoe.


# Detect gamestate/changes in gamestate 

# Detect if game has ended: win/loss/draw

# Return 

def isGameOver(state):
    '''
    returns whether the is game is not over. 
    '''

    if getWinner():
        return True

    else:
        return False
        
def getWinningThree(state):
    '''
    returns a 3x1 array of the board indices for the winning 3 in a row
    if there is no 3 in a row, returns None
    '''

    wins = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], #rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], #cols
        [0, 4, 8], [2, 4, 6] #diagonals
    ]

    # check if any row, col, or diagonal has three 'X's or 'O's 
    for w in wins:
        first, second, third = w

        if state[first] == state[second] == state[third] and state[first] != None:
            return w

    # no valid three in a row was found
    return None


def getWinner(state):
    '''
    if the game is not over, return None
    return 'robot' if the robot wins, based on whether there are 3 'X's in a row on the board
    return 'human' if the human wins, based on whether there are 3 'O's in a row on the board
    return 'draw' if neither robot or human won
    '''

    win = getWinningThree(state)
        
    if win and state[win[0]] == 'X': 
        return "robot"

    elif win and state[win[0]] == 'O': 
        return "human"

    elif getPossibleMoves(state) == 0:
        return "draw"

    else:
        return None


def getPossibleMoves(state):
    '''
    return an array of all the possible moves a robot can take
    if there are no possible moves return empty array, this means our board is full and the game is over
    '''
    return [i for i in range(len(state)) if state[i] == None]

def getOptimalMove(state):
    '''
    return index into board of optimal move for robot to make
    assumes the game is not over 
    '''
    #TODO
    #start with random
    pass 

def processBoard():
    '''
    
    '''
    # TODO: get camera working
    cam = cv2.VideoCapture(0)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Preprocess the frame (grayscale, blur, threshold, etc.).
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Display the resulting frame
        cv2.imshow('frame', gray)

        #
        if cv2.waitKey(10) == ord('q'):
            break

        # Detect the grid lines.

        
    # Release the video capture and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()



def main():
    #create an array to hold the gamestate
    gamestate = np.array([None,None,None,
                        None, None,None,
                        None,None,None])

    # Read in video feed 
    processBoard()
    
    '''
    if board is done drawing:
        while not isGameOver(state): 
            
    '''

