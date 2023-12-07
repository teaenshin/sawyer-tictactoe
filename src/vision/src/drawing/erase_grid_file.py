#!/usr/bin/env python
import rospy
import tf2_ros
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
import numpy as np
from numpy import linalg
import sys
from drawing import draw_grid_file
from drawing import joint_angles
# import draw_grid_file
# import joint_angles

# SET THIS BEFOREHAND AND UPDTAE IN DRAW_X_FILE and DRAW_WINFILE
# z = draw_grid_file.z + 0.022 #0.070
z = -0.120 - 0.006

def erase_grid():
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
    input('Press [ Enter ] to start erasing: Make sure eraser is attached.')
    
    # Construct the request
    request = GetPositionIKRequest()
    request.ik_request.group_name = draw_grid_file.GROUP_NAME

    # If a Sawyer does not have a gripper, replace '_gripper_tip' with '_wrist' instead
    link = draw_grid_file.LINK #"right_gripper_tip"

    request.ik_request.ik_link_name = link
    request.ik_request.pose_stamped.header.frame_id = "base"
    request.ik_request.pose_stamped.pose.orientation.x = 0
    request.ik_request.pose_stamped.pose.orientation.y = 1
    request.ik_request.pose_stamped.pose.orientation.z = 0
    request.ik_request.pose_stamped.pose.orientation.w = 0

    width = 0.23
    offs = 0.15/3
    # height= 0.02
    
    try:
        trans = tfBuffer.lookup_transform("base", link, rospy.Time())   # TODO: may need to update frames
        # trans.transform.translation gives current x, y, z
        
        # find curr location of end effector:
        # TODO
        # x = trans.transform.translation.x
        # y = trans.transform.translation.y
        x = joint_angles.CUSTOM_TUCK[0] + offs
        y = joint_angles.CUSTOM_TUCK[1]+0.015
        
        # trans.transform.translation gives current x, y, z
        locs = [(x, y, z), (x, y-width/3, z), (x, y-2*width/3, z), (x, y-width, z),(x, y-width-0.03, z), (x, y-width-0.03, z+0.02), (x+offs, y-width-0.03, z+0.02), (x+offs, y-width-0.03, z), # border horizontal
                (x+offs, y-width, z), (x+offs, y-2*width/3, z), (x+offs, y-width/3, z), (x+offs, y, z),(x+offs, y, z), (x+offs, y, z+0.02), (x+(2*offs), y, z+0.02),(x+(2*offs), y, z), #horizontal 1
                (x+(2*offs), y, z), (x+(2*offs), y-width/3, z), (x+(2*offs), y-2*width/3, z), (x+(2*offs), y-width, z),(x+(2*offs), y-width-0.03, z), (x+(2*offs), y-width-0.03, z+0.02), (None, None, None), (x+(3*offs), y-width-0.03, z+0.02),(x+(3*offs), y-width-0.03, z),  #horizontal 2
                (x+(3*offs), y-width, z), (x+(3*offs), y-2*width/3, z), (x+(3*offs), y-width/3, z), (x+(3*offs), y, z), (x+(3*offs), y, z), (x+(3*offs), y, z+0.02) #(x+width, y, z),(x+width, y-offs, z), # border horizontal
                ]

        # locs = [(x, y, z), (x, y-width/3, z), (x, y-2*width/3, z), (x, y-width, z),(x, y-width, z),(x+offs, y-width, z), # border horizontal
        #         (x+offs, y-width, z), (x+offs, y-2*width/3, z), (x+offs, y-width/3, z), (x+offs, y, z),(x+offs, y, z),(x+(2*offs), y, z), #horizontal 1
        #         (x+(2*offs), y, z), (x+(2*offs), y-width/3, z), (x+(2*offs), y-2*width/3, z), (x+(2*offs), y-width, z),(x+(2*offs), y-width, z),(x+(3*offs), y-width, z), #horizontal 2
        #         (x+(3*offs), y-width, z), (x+(3*offs), y-2*width/3, z), (x+(3*offs), y-width/3, z), (x+(3*offs), y, z), (x+(3*offs), y, z) #(x+(4*offs), y-width, z),  # border horizontal

        #         # (x+(4*offs), y, z), (x+(4*offs), y-width/3, z), (x+(4*offs), y-2*width/3, z), (x+(4*offs), y-width, z),(x+(4*offs), y-width, z),(x+(5*offs), y-width, z), # border horizontal
        #         # (x+(5*offs), y-width, z), (x+(5*offs), y-2*width/3, z), (x+(5*offs), y-width/3, z), (x+(5*offs), y, z),(x+(5*offs), y, z),(x+(6*offs), y, z), #horizontal 1
        #         # (x+(6*offs), y, z), (x+(6*offs), y-width/3, z), (x+(6*offs), y-2*width/3, z), (x+(6*offs), y-width, z),(x+(6*offs), y-width, z) #horizontal 2
        #         ]
        
        for x1, y1, z1 in locs:

            if x1 is None:
                joint_angles.main()
                continue
                
            # Set the desired orientation for the end effector HERE (marker touches board)
            change = z1 - request.ik_request.pose_stamped.pose.position.z
            request.ik_request.pose_stamped.pose.position.x = x1
            request.ik_request.pose_stamped.pose.position.y = y1
            request.ik_request.pose_stamped.pose.position.z = z1         

            # Send the request to the service
            response = compute_ik(request)
            
            # Print the response HERE
            print(response)
            group = MoveGroupCommander(draw_grid_file.GROUP_NAME)

            # Setting position and orientation target
            group.set_pose_target(request.ik_request.pose_stamped)
            group.limit_max_cartesian_link_speed(0.8)
            group.set_max_velocity_scaling_factor(1)

            # Plan IK
            plan = group.plan()
            
            
            # # Execute IK if safe

            if (change != 0):
                # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")

                # while user_input == 'n':
                #     plan = group.plan()
                #     user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")

                # if user_input == 'y':
                group.execute(plan[1])

            else:

                # if user_input == 'y':
                group.execute(plan[1])  

        joint_angles.main()

        # go to default position
        # joint_angles.main()


        # for x1, y1, z1 in [(x+width, y, z), (x+width, y, z)] + locs[::-1] + [(0.694, 0.158, 0.525)]: # last coord is tuck
                
        #     # Set the desired orientation for the end effector HERE (marker touches board)
        #     change = z1 - request.ik_request.pose_stamped.pose.position.z
        #     request.ik_request.pose_stamped.pose.position.x = x1
        #     request.ik_request.pose_stamped.pose.position.y = y1
        #     request.ik_request.pose_stamped.pose.position.z = z1      

        #     # move robot to loc 1
        #     # Send the request to the service
        #     response = compute_ik(request)
            
        #     # Print the response HERE
        #     print(response)
        #     group = MoveGroupCommander("right_arm")

        #     # Setting position and orientation target
        #     group.set_pose_target(request.ik_request.pose_stamped)

        #     # Set max speed
        #     group.limit_max_cartesian_link_speed(0.2)

        #     # Plan IK
        #     plan = group.plan()
        #     # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
        #     # # Execute IK if safe
        #     # # if user_input == 'y':
        #     # #     group.execute(plan[1])

        #     # while user_input == 'n':
        #     #     plan = group.plan()
        #     #     user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")

            
        #     # if user_input == 'y':
        #     # group.execute(plan[1])  
        #     # if (change != 0):
        #     #     user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")

        #     #     while user_input == 'n':
        #     #         plan = group.plan()
        #     #         user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")

        #     #     if user_input == 'y':
        #     #         group.execute(plan[1])

        #     # else:

        #     #     # if user_input == 'y':
        #     group.execute(plan[1])  
        
        
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


# Python's syntax for a main() method
if __name__ == '__main__':
    rospy.init_node('service_query')
    erase_grid()

