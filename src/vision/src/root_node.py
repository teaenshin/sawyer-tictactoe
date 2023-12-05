#!/usr/bin/env python
import numpy as np
import rospy
import sys
from vision_utils import *
from utils import *

from vision.msg import BoardData
from vision.srv import Robot

# TODO: clean up code in this file

class RootNode:
    ''' Subscriber for gamestate'''
    def __init__(self) -> None:
        rospy.init_node('root_node')
        self.subscriber = rospy.Subscriber("board_data_topic", BoardData, self.gamestate_callback)
        self.gamestate = None
        self.game_over = False
        self.robot_client(2, None, None) # draw grid


    # TODO (maybe), it might happen that the publisher publishes a message while the callback is executing. We don't want to process this
    # In that case we would add a timer and ensure there is a minimum time buffer between callbacks
    def gamestate_callback(self, data):
        if self.game_over:
            return
        cur_gamestate = np.array(data.data)
        rospy.loginfo(rospy.get_caller_id() + " received data: %s", cur_gamestate)

        change = None 
        if self.gamestate is None:
            self.gamestate = cur_gamestate 

        elif self.gamestate is not None and not np.array_equal(self.gamestate, cur_gamestate):
            # change in gamestate detected
            rospy.loginfo("Detected change in gamestate!")
            for i in range(9):
                if self.gamestate[i] is None or self.gamestate[i] != cur_gamestate[i]:
                    self.gamestate[i] = cur_gamestate[i]
                    if change:
                        print("More than 1 changes detected in gamestate...somethign is weird. Just gonna pick last change")
                    change = self.gamestate[i]
        if change:
            self.update(change)

    # player is either X or O and depending on which one it is we act accordingly
    def update(self, player):
        ''' player is the player that made the change last'''
        if check_win(self.gamestate):
            self.game_over = True
            if check_win(self.gamestate) == "X":
                win = self.getWinningThree()
                self.robot_client(1, None, win)
        elif check_draw(self.gamestate):
            self.game_over = True
            print("DRAW DETECTED")
            # TODO: publish something 
            
        else:
            if player == "O": # if human just went
                move = self.pick_move()
                print(f"Robot should draw at cell {move}")
                self.robot_client(0, 8-move, None)
        
    def pick_move(self):
        
        # If any spot either gives you the win, or would give the other player the win, pick it
        for i in range(9):
            if self.gamestate[i] == "":
                temp_gamestate = self.gamestate.copy()
                temp_gamestate[i] = "X"
                if check_win(temp_gamestate):
                    return i
                temp_gamestate[i] = "O"
                if check_win(temp_gamestate):
                    return i
                
        # If center is still open go center this will happen in the first two turns for sure
        if self.gamestate[4] == "":
            return 4
        
        corners = [0, 2, 6, 8]
        for corner in corners:
            if self.gamestate[corner] == "":
                return corner
        
        # If neither of those pick the first open spot
        #TODO better strategy
        for i in range(9):
            if self.gamestate[i] == "":
                return i
            

    def getWinningThree(self):
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

            if self.gamestate[first] == self.gamestate[second] == self.gamestate[third] and self.gamestate[first] != None:
                return [8-x for x in w]

        # no valid three in a row was found
        return None



    def robot_client(self, type, idx, win):
        rospy.wait_for_service('/draw_service')
        print("service is ready")

        try: 
            robot_proxy = rospy.ServiceProxy('/draw_service', Robot)
            robot_proxy(type, idx, win)

        except rospy.ServiceException as e:
            rospy.loginfo(e)


    def main(self):
        print("START ROOT_NODE")
        rospy.wait_for_service('/draw_service')
        print("/DRAW_SERVICE IS READY")
        while not rospy.is_shutdown() and not self.game_over:
            print("GAME IS OVER")
            self.robot_client(3, None, None) # erase board
            rospy.sleep(0.1)

        # Perform any cleanup if necessary before exiting
        rospy.loginfo("Exiting root_node")

if __name__ == '__main__':
    root = RootNode()
    root.main()
