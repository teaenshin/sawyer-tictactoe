#!/usr/bin/env python
import rospy
from vision.srv import Robot
from drawing import draw_x_file
from drawing import draw_win_file
from drawing import draw_grid_file
from drawing import erase_grid_file

def robot_callback(request):
    print("Received request type: %s" % request.type)

    if request.type == 0:
        idx = request.index
        draw_x_file.draw_x(idx)

    if request.type == 1:
        win = request.win
        draw_win_file.draw_win(win)


    if request.type == 2:
        draw_grid_file.draw_grid()

    if request.type == 3:
        erase_grid_file.erase_grid()

    # response = "Received: " + str(req.data)
    # return DRAW_RESPONSE(True, response)
    return []

def robot_server():
    print("START ROBOT_SERVER")
    rospy.init_node('robot_server')
    print("ROBOT_SERVER INITED")
    s = rospy.Service('/draw_service', Robot, robot_callback)
    rospy.spin()

if __name__ == "__main__":
    robot_server()
