#!/usr/bin/env python
import numpy as np
import rospy
import sys
from vision_utils import *
from utils import *

from vision.msg import BoardData

# Import definition of DRAW REQUESt (srv)

class RootNode:
    ''' Subscriber for gamestate'''
    def __init__(self) -> None:
        rospy.init_node('root_node')
        self.subscriber = rospy.Subscriber("board_data_topic", BoardData, self.gamestate_callback)
        self.gamestate = None
        self.game_over = False


    # TODO (maybe), it might happen that the publisher publishes a message while the callback is executing. We don't want to process this
    # In that case we would add a timer and ensure there is a minimum time buffer between callbacks
    def gamestate_callback(self, data):
        cur_gamestate = np.array(data.data)
        rospy.loginfo(rospy.get_caller_id() + " received data: %s", cur_gamestate)

        change = None 
        if self.gamestate is None:
            self.gamestate = cur_gamestate 

        elif self.gamestate is not None and not np.array_equal(self.gamestate, cur_gamestate):
            # change in gamestate detected
            rospy.loginfo("Detected change in gamestate!")
            # TODO: check that there is only 1 change
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
                #TODO: publish game over to motion node and draw line
                pass
        elif  check_draw(self.gamestate):
            self.game_over = True
            print("DRAW DETECTED")
            # TODO: publish something 
            pass
        else:
            if player == "O": # if human just went
                move = self.pick_move()
                print(f"Robot should draw at cell {move}")
                #TODO publish where to draw  X
        
    def pick_move(self):
        # If center is still open go center this will happen in the first two turns for sure
        if self.gamestate[4] == "":
            return 4
        # Otherwise loop through all open spots
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
        
        # If neither of those pick the first open spot
        #TODO better strategy
        for i in range(9):
            if self.gamestate[i] == "":
                return i


    # def call_draw_service(self, x):
    #     rospy.wait_for_service('/draw')
    #     try:
    #         service = rospy.ServiceProxy('/draw', DRAW_REQUEST)
    #         resp = service(x)
    #         return resp.success, resp.message
    #     except rospy.ServiceException as e:
    #         rospy.logerr("Service call failed: %s", e)


    def main(self):
        while not rospy.is_shutdown() and not self.game_over:
            rospy.sleep(0.1)

        # Perform any cleanup if necessary before exiting
        rospy.loginfo("Exiting the node")

if __name__ == '__main__':
    root = RootNode()
    root.main()
