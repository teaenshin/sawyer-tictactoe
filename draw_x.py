# PACKAGE OF FILE SHOULD INCLUDE intera_interface, rospy and std_msgs
# (also need planning package from lab 5)

# Current implementation: Draws an X at the current location

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

# row_coord = [_, _, _] # TODO: x-coord corresponding to each row
# col_coord = [_, _, _] # TODO: y-coord corresponding to each col

def main():
    # Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    rospy.init_node('service_query')
    # Create the function used to call the service
    compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)

    # Set up the tfBuffer
    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer)
    r = rospy.Rate(1)

    height = 0.5 # TODO: adjust 
    x_width = 0.5 # TODO: adjust 

    # drawing visual:
    # (1) \     / (4)(5)
    #       \ /
    #       / \
    # (6) /     \ (2)(3)
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
        request.ik_request.pose_stamped.pose.orientation.x = 0.0
        request.ik_request.pose_stamped.pose.orientation.y = 1.0
        request.ik_request.pose_stamped.pose.orientation.z = 0.0
        request.ik_request.pose_stamped.pose.orientation.w = 0.0

        try:
            # find curr location of end effector:
            trans = tfBuffer.lookup_transform("base", "right_hand", rospy.Time())   # TODO: may need to update frames
            # trans.transform.translation gives current x, y, z

            # Set the desired orientation for the end effector HERE (marker touches board)
            request.ik_request.pose_stamped.pose.position.x = trans.transform.translation.x
            request.ik_request.pose_stamped.pose.position.y = trans.transform.translation.y
            request.ik_request.pose_stamped.pose.position.z = trans.transform.translation.z - height      

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
            
            # Execute IK if safe
            if user_input == 'y':
                group.execute(plan[1])

            # move robot to loc 2 (draw first line of X)
            request.ik_request.pose_stamped.pose.position.x += x_width # TODO: adjust
            request.ik_request.pose_stamped.pose.position.y -= x_width # TODO: adjust

            # Send the request to the service
            response = compute_ik(request)
            
            # Print the response HERE
            print(response)
            group = MoveGroupCommander("right_arm")

            # Setting position and orientation target
            group.set_pose_target(request.ik_request.pose_stamped)

            # Plan IK
            plan = group.plan()
            # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # Execute IK if safe
            # if user_input == 'y':
            group.execute(plan[1])

            # move robot to loc 3 (lift marker)
            request.ik_request.pose_stamped.pose.position.z += height

            # Send the request to the service
            response = compute_ik(request)
            
            # Print the response HERE
            print(response)
            group = MoveGroupCommander("right_arm")

            # Setting position and orientation target
            group.set_pose_target(request.ik_request.pose_stamped)

            # Plan IK
            plan = group.plan()
            # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # Execute IK if safe
            # if user_input == 'y':
            group.execute(plan[1])

             # move robot to loc 4 (location of second line)
            request.ik_request.pose_stamped.pose.position.y += x_width

            # Send the request to the service
            response = compute_ik(request)
            
            # Print the response HERE
            print(response)
            group = MoveGroupCommander("right_arm")

            # Setting position and orientation target
            group.set_pose_target(request.ik_request.pose_stamped)

            # Plan IK
            plan = group.plan()
            # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # Execute IK if safe
            # if user_input == 'y':
            group.execute(plan[1])

             # move robot to loc 5 (marker touches board)
            request.ik_request.pose_stamped.pose.position.z -= height

            # Send the request to the service
            response = compute_ik(request)
            
            # Print the response HERE
            print(response)
            group = MoveGroupCommander("right_arm")

            # Setting position and orientation target
            group.set_pose_target(request.ik_request.pose_stamped)

            # Plan IK
            plan = group.plan()
            # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # Execute IK if safe
            # if user_input == 'y':
            group.execute(plan[1])

             # move robot to loc 6 (draw second line of X)
            request.ik_request.pose_stamped.pose.position.x -= x_width
            request.ik_request.pose_stamped.pose.position.y -= x_width

            # Send the request to the service
            response = compute_ik(request)
            
            # Print the response HERE
            print(response)
            group = MoveGroupCommander("right_arm")

            # Setting position and orientation target
            group.set_pose_target(request.ik_request.pose_stamped)

            # Plan IK
            plan = group.plan()
            # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # Execute IK if safe
            # if user_input == 'y':
            group.execute(plan[1])

            # TODO: move robot back to tuck???
            
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)


# Python's syntax for a main() method
if __name__ == '__main__':
    main()

