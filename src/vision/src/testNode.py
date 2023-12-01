#!/usr/bin/env python
import numpy as np
import rospy
import sys
from vision_utils import *
from utils import *

from vision.msg import BoardData
from vision.srv import Robot

def robot_client(self):
    rospy.init_node('robot_client')
    rospy.wait_for_service('/draw_service')

    try: 
        robot_proxy = rospy.ServiceProxy('/draw_service', Robot)

        # win = self.getRobotWin(self.gamestate)

        # # robot won -> robot draws win line
        # if win is not None:
        #     robot_proxy(1, None, win)

        # # human made move -> robot draws x
        # elif not self.getWinner() and self.updated:  # check that human made move and the game is not over
        #     idx = self.pick_move()
        #     self.updated = False
        #     robot_proxy(0, idx, None)

        # else:
        #     robot_proxy(-1, None, None)

        robot_proxy(-1, None, None)


    except rospy.ServiceException as e:
        rospy.loginfo(e)