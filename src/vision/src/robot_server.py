#!/usr/bin/env python
import rospy
from vision.srv import Robot
from sawyer.src.draw_x import draw_x
from sawyer.src.draw_win import draw_win


#TODO define what a draw request (srv) is and a draw response (msg) is


def robot_callback(request):
    print("Received request type: %s" % request.type)
    #TODO upon recieving a draw request call the code to actually draw the X or line

    if request.type == 0:
        idx = request.index
        draw_x(idx)

    if request.type == 1:
        win = request.win
        draw_win(win)

    # response = "Received: " + str(req.data)
    # return DRAW_RESPONSE(True, response)

def robot_server():
    rospy.init_node('robot_server')
    s = rospy.Service('/draw_service', Robot, robot_callback)
    rospy.spin()

if __name__ == "__main__":
    robot_server()
