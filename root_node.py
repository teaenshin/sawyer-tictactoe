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
        rospy.Subscriber("board_data_topic", BOARD_DATA, self.board_callback)
        self.board = None
        self.game_over = False

    # TODO (maybe), it might happen that the publisher publishes a message while the callback is executing. We don't want to process this
    # In that case we would add a timer and ensure there is a minimum time buffer between callbacks
    def board_callback(self, board):
        rospy.loginfo(rospy.get_caller_id() + " received data: %s", board.data)
        updated = None
        if self.board is None:
            self.board = board.data
        else:
            new_board = board.data
            #loop through board and if a previously empty spot is now filled update
            for spot in range(9):
                if not self.board[spot] and new_board[spot]:
                    self.board[spot] = new_board[spot]
                    updated = new_board[spot]
        
        if updated:
            self.update(updated)

    # player is either X or O and depending on which one it is we act accordingly
    def update(self, player):
        if utils.check_win(self.board) or utils.check_draw(self.board):
            self.game_over = True
            if utils.check_win(self.board) == "X":
                #TODO call draw line here
                pass
        else:
            if player == "O":
                move = self.pick_move()
                #TODO call the draw x function here using move
        
    def pick_move(self):
        # If center is still open go center this will happen in the first two turns for sure
        if self.board[4] == "":
            return 4
        # Otherwise loop through all open spots
        # If any spot either gives you the win, or would give the other player the win, pick it
        for i in range(9):
            if self.board[i] == "":
                temp_board = self.board.copy()
                temp_board[i] = "X"
                if utils.check_win(temp_board):
                    return i
                temp_board[i] = "O"
                if utils.check_win(temp_board):
                    return i
        
        # If neither of those pick the first open spot
        #TODO better strategy
        for i in range(9):
            if self.board[i] == "":
                return i


    def call_draw_service(self, x):
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
