#!/usr/bin/env python
import rospy
# from std_msgs.msg import Int16MultiArray
import tf2_ros
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
import numpy as np
from numpy import linalg
import sys
from drawing import draw_grid_file
import joint_angles

# SET BEFOREHAND
z = draw_grid_file.Z

tuck = (0.694, 0.158, 0.525)
row_coord = [tuck[0] + 0.02 + 2 * 0.2/3, tuck[0] + 0.02 + 0.2/3, tuck[0] + 0.02] 
col_coord = [tuck[1] - 0.02 , tuck[1] - 0.02 - 0.2/3, tuck[1] - 0.02 - 2 * 0.2/3]

def draw_x(msg):
    print("hello")
    # Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    # rospy.init_node('service_query')
    # Create the function used to call the service
    compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)

    # Set up the tfBuffer
    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer)
    r = rospy.Rate(1)

    height = 0.02 # TODO: adjust 
    x_width = 0.03 # TODO: adjust 

    # drawing visual:
    # (1) \     / (4)(5)
    #       \ /
    #       / \
    # (6) /     \ (2)(3)
    # while not rospy.is_shutdown():
    # input('Press [ Enter ]: ')
    
    # Construct the request
    request = GetPositionIKRequest()
    request.ik_request.group_name = draw_grid_file.GROUP_NAME

    # If a Sawyer does not have a gripper, replace '_gripper_tip' with '_wrist' instead
    link = draw_grid_file.LINK

    request.ik_request.ik_link_name = link
    # request.ik_request.attempts = 20
    request.ik_request.pose_stamped.header.frame_id = "base"
    request.ik_request.pose_stamped.pose.orientation.x = 0.0
    request.ik_request.pose_stamped.pose.orientation.y = 1.0
    request.ik_request.pose_stamped.pose.orientation.z = 0.0
    request.ik_request.pose_stamped.pose.orientation.w = 0.0

    z0 = 0.3
    x = row_coord[msg //3]
    y = col_coord[msg %3]

    try:

        locs = [(x, y, z0), (x, y, z), (x + x_width, y - x_width, z), (x + x_width, y - x_width, z + height), (x + x_width, y, z + height), (x + x_width, y, z), (x, y - x_width, z), (0.694, 0.158, 0.525)]

        for x1, y1, z1 in locs:

            #move arm to correct grid spot
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


            # Plan IK
            plan = group.plan()
            # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # # Execute IK if safe
            # if user_input == 'y':
            group.execute(plan[1])

        # go to default position
        # joint_angles.main()

        
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


# Python's syntax for a main() method
if __name__ == '__main__':
    rospy.init_node('service_query')
    draw_x(0)

