#!/usr/bin/env python
import numpy as np
import rospy
import sys
import utils

#TODO define BOARD DATA (msg)
# Import definition of DRAW REQUESt (srv)

class RootNode:
    def __init__(self) -> None:
        rospy.init_node('root_node')
        rospy.Subscriber("vision_node", BOARD_DATA, self.board_callback)
        self.board = None
        self.game_over = False


    def board_callback(self, board):
        rospy.loginfo(rospy.get_caller_id() + " received data: %s", board.data)
        updated = None
        if self.board is None:
            self.board = board.data
        else:
            new_board = board.data
            #loop through board and if a previously empty spot is now filled update
            for row in range(2):
                for col in range(2):
                    if not self.board[row][col] and new_board[row][col]:
                        self.board[row][col] = new_board[row][col]
                        updated = new_board[row][col]
        
        if updated:
            self.update(new_board[row][col])

    # player is either X or O and depending on which one it is we act accordingly
    def update(self, player):
        if utils.check_win(self.board) or utils.check_draw(self.board):
            self.game_over = True
            if utils.check_win(self.board) == "X":
                #TODO call draw line here
                pass
        else:
            if player == "O":
                #TODO call the regular turn functionality here
                pass
        


    def call_service(self, x):
        rospy.wait_for_service('/draw')
        try:
            service = rospy.ServiceProxy('/draw', DRAW_REQUEST)
            resp = service(x)
            return resp.success, resp.message
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)


    def main(self):
        while not rospy.is_shutdown() and not self.game_over:
            rospy.sleep(0.1)

        # Perform any cleanup if necessary before exiting
        rospy.loginfo("Exiting the node")

if __name__ == '__main__':
    root = RootNode()
    root.main()
