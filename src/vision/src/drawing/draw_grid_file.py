#!/usr/bin/env python
import rospy
import tf2_ros
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
import numpy as np
from numpy import linalg
import sys
import intera_interface
from drawing import joint_angles

# SET THIS BEFOREHAND AND UPDTAE IN DRAW_X_FILE and DRAW_WINFILE
z = -0.152 - 0.0065 
GROUP_NAME = 'right_arm'
LINK = "right_gripper_tip"

def draw_grid():
    # Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    # Create the function used to call the service
    compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)

    # Set up the tfBuffer
    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer)
    r = rospy.Rate(1)


# drawing visual:
#         (1)|  | (6)(7)
# (10)(11) --|--|-- (8)(9)
# (12)(13) --|--|-- (14)
#         (2)|  | (4)
#         (3)     (5)

    # while not rospy.is_shutdown():
    input('Press [ Enter ] to draw grid: ')

    rp = intera_interface.RobotParams()
    validLimbs = rp.get_limb_names()
    print(validLimbs)

    # limb = intera_interface.Limb("righ")

    
    # Construct the request
    request = GetPositionIKRequest()
    request.ik_request.group_name = GROUP_NAME

    # If a Sawyer does not have a gripper, replace '_gripper_tip' with '_wrist' instead
    link = LINK #"right_gripper_tip"

    request.ik_request.ik_link_name = link
    request.ik_request.pose_stamped.header.frame_id = "base"
    request.ik_request.pose_stamped.pose.orientation.x = 0
    request.ik_request.pose_stamped.pose.orientation.y = 1
    request.ik_request.pose_stamped.pose.orientation.z = 0
    request.ik_request.pose_stamped.pose.orientation.w = 0

    width = 0.2
    offs = width/3
    height = 0.04
    
    try:
        # trans = tfBuffer.lookup_transform("base", link, rospy.Time())   # TODO: may need to update frames
        # trans.transform.translation gives current x, y, z
        # x = trans.transform.translation.x
        # y = trans.transform.translation.y
        # print('x, y before', x, y)
        joint_angles.main()
        # x = 0.694
        # y = 0.158

        x = 0.613 # this is the x, y position of our custom tuck  # joint_angles.CUSTOM_TUCK[0]
        y = 0.154 # joint_angles.CUSTOM_TUCK[1]
        # find curr location of end effector:
        
        # trans.transform.translation gives current x, y, z
        
        locs = [(x, y, z), (x, y-width/3, z), (x, y-2*width/3, z), (x, y-width, z),(x, y-width, z+height),(x+offs, y-width, z+height), # border horizontal
                (x+offs, y-width, z), (x+offs, y-2*width/3, z), (x+offs, y-width/3, z), (x+offs, y, z),(x+offs, y, z+height),(x+(2*offs), y, z+height), #horizontal 1
                (x+(2*offs), y, z-0.002), (x+(2*offs), y-width/3, z-0.002), (x+(2*offs), y-2*width/3, z-0.002), (x+(2*offs), y-width, z-0.002),(x+(2*offs), y-width, z+height),(x+width, y-width, z+height), #horizontal 2
                (x+width, y-width, z-0.002), (x+width, y-2*width/3, z-0.002), (x+width, y-width/3, z-0.002), (x+width, y, z-0.002), (x+width, y, z+height) #(x+width, y, z+height),(x+width, y-offs, z+height), # border horizontal
                ]
        
        
        # Draw horizontal Lines
        '''
        for x1, y1, z1 in locs:
                
                # Set the desired orientation for the end effector HERE (marker touches board)
            change = z1 - request.ik_request.pose_stamped.pose.position.z
            request.ik_request.pose_stamped.pose.position.x = x1
            request.ik_request.pose_stamped.pose.position.y = y1
            request.ik_request.pose_stamped.pose.position.z = z1 # TODO: adjust        

            # move robot to loc 1
            # Send the request to the service
            response = compute_ik(request)
            
            # Print the response HERE
            print(response)
            group = MoveGroupCommander(GROUP_NAME)
            group.limit_max_cartesian_link_speed(0.8)

            # Setting position and orientation target
            group.set_pose_target(request.ik_request.pose_stamped)

            # Plan IK
            plan = group.plan()
            
            
            # # Execute IK if safe

            # if (change != 0):
            #     user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")

            #     while user_input == 'n':
            #         plan = group.plan()
            #         user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")

            #     if user_input == 'y':
            #         group.execute(plan[1])

            # else:

            #     # if user_input == 'y':
            group.execute(plan[1])  
        '''
        input("Press [Enter] to continue drawing vertical lines. Sawyer will tuck to custom psoition. ")

        #$ move gripper behind board
        joint_angles.main()

        vertLocs = [(x, y, z), (x+width/3, y, z), (x+2*width/3, y, z), (x+width, y, z), (x+width, y, z+height), (None, None, None),
                    (x, y-width/3, z+height), (x, y-width/3, z), (x+width/3, y-width/3, z), (x+2*width/3, y-width/3, z), (x+width, y-width/3, z), (x+width, y-width/3, z+height),  (None, None, None),
                    (x, y-2*width/3, z+height), (x, y-2*width/3, z), (x+width/3, y-2*width/3, z), (x+2*width/3, y-2*width/3, z), (x+width, y-2*width/3, z), (x+width, y-2*width/3, z+height),  (None, None, None),
                    (x, y-width, z+height), (x, y-width, z), (x+width/3, y-width, z), (x+2*width/3, y-width, z), (x+width, y-width, z), (x+width, y-width, z+height)
                ]

        # for x1, y1, z1 in [(x+width, y, z+height), (x+width, y, z+height)] + locs[::-1] + [(0.694, 0.158, 0.525)]: # last coord is tuck

        for x1, y1, z1 in vertLocs: # last coord is tuck
            if x1 is None:
                joint_angles.main()
                continue

            # Set the desired orientation for the end effector HERE (marker touches board)
            change = z1 - request.ik_request.pose_stamped.pose.position.z
            request.ik_request.pose_stamped.pose.position.x = x1
            request.ik_request.pose_stamped.pose.position.y = y1
            request.ik_request.pose_stamped.pose.position.z = z1      

            # move robot to loc 1
            # Send the request to the service
            response = compute_ik(request)
            
            # Print the response HERE
            print(response)
            group = MoveGroupCommander(GROUP_NAME)
            group.limit_max_cartesian_link_speed(0.8)


            # Setting position and orientation target
            group.set_pose_target(request.ik_request.pose_stamped)

            # Plan IK
            plan = group.plan()            

            group.execute(plan[1])

            # go to default position
        joint_angles.main()
           
            
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


# Python's syntax for a main() method
if __name__ == '__main__':
    rospy.init_node('service_query')
    draw_grid()

