#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16MultiArray
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
import numpy as np
from numpy import linalg
import sys

# TODO: modify so that it's the center coords
# row_coord = [0.839, 0.775, 0.710]
# col_coord = [0.141, 0.077, 0.012]

tuck = (0.694, 0.158, 0.525)
row_coord = [tuck[0] + 5 * 0.2/6, tuck[0] + 3 * 0.2/6, tuck[0] + 0.2/6] # TODO: x-coord corresponding to each row
col_coord = [tuck[1] - 0.2/6 , tuck[1] - 3 * 0.2/6, tuck[1] - 5 * 0.2/6] # TODO: y-coord corresponding to each col

# # #
# # #
# # #

# column: index % 3
# row: index // 3

def draw_win(msg):
    # Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    rospy.init_node('service_query')
    # Create the function used to call the service
    compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
    while not rospy.is_shutdown():
        input('Press [ Enter ]: ')
        
        # Construct the request
        request = GetPositionIKRequest()
        request.ik_request.group_name = "right_arm"

        # If a Sawyer does not have a gripper, replace '_gripper_tip' with '_wrist' instead
        link = "right_gripper_tip"

        request.ik_request.ik_link_name = link
        # request.ik_request.attempts = 20
        request.ik_request.pose_stamped.header.frame_id = "base"
        
        # Set the desired orientation for the end effector HERE 
        request.ik_request.pose_stamped.pose.orientation.x = 0.0
        request.ik_request.pose_stamped.pose.orientation.y = 1.0
        request.ik_request.pose_stamped.pose.orientation.z = 0.0
        request.ik_request.pose_stamped.pose.orientation.w = 0.0

        z = 0.019
        
        try:

            locs = [(row_coord[msg[0]//3], col_coord[msg[0]%3], z), (row_coord[msg[1]//3], col_coord[msg[1]%3], z), (row_coord[msg[2]//3], col_coord[msg[2]%3], z), (0.694, 0.158, 0.525)]

            for x1, y1, z1 in locs:
                    
                    # Set the desired orientation for the end effector HERE (marker touches board)
                request.ik_request.pose_stamped.pose.position.x = x1
                request.ik_request.pose_stamped.pose.position.y = y1
                request.ik_request.pose_stamped.pose.position.z = z1       

                # move robot to loc 1
                # Send the request to the service
                response = compute_ik(request)
                
                # Print the response HERE
                print(response)
                group = MoveGroupCommander("right_arm")

                # Setting position and orientation target
                group.set_pose_target(request.ik_request.pose_stamped)

                # Plan IK
                plan = group.plan()
                user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
                
                # # Execute IK if safe
                # # if user_input == 'y':
                # #     group.execute(plan[1])

                # while user_input == 'n':
                #     plan = group.plan()
                #     user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")

                if user_input == 'y':
                    group.execute(plan[1])  
            
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)


def listener():

    # Create a new instance of the rospy.Subscriber object which we can use to
    # receive messages of type std_msgs/String from the topic /chatter_talk.
    # Whenever a new message is received, the method callback() will be called
    # with the received message as its first argument.
    rospy.Subscriber("grid_index", Int16MultiArray, callback)

    # Wait for messages to arrive on the subscribed topics, and exit the node
    # when it is killed with Ctrl+C
    rospy.spin()


# Python's syntax for a main() method
if __name__ == '__main__':
    
    # Run this program as a new node in the ROS computation graph called
    # /listener_<id>, where <id> is a randomly generated numeric string. This
    # randomly generated name means we can start multiple copies of this node
    # without having multiple nodes with the same name, which ROS doesn't allow.
    # rospy.init_node('listener', anonymous=True) # ??

    # listener()
    draw_win()


# lab5/src/move_arm/src/ik_example.py
# lab2/src/my_chatter/src/sub.py
