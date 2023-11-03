# PACKAGE OF FILE SHOULD INCLUDE intera_interface, rospy and std_msgs
# (also need planning package from lab 5)

#!/usr/bin/env python
import rospy
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
import numpy as np
from numpy import linalg
import sys

def main():
    # Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    rospy.init_node('service_query')
    # Create the function used to call the service
    compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)

# drawing visual:
#         (1)|  | (6)(7)
# (10)(11) --|--|-- (8)(9)
# (12)(13) --|--|-- (14)
#         (2)|  | (4)
#         (3)     (5)

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

        # find curr location of end effector:
        x = _   # TODO: grid loc x
        y = _   # TODO: grid loc y
        z = _   # TODO: grid loc z (marker lifted)

        width = 3
        offs = width/3
        height = 0.5
        # trans.transform.translation gives current x, y, z
        locs = [(x, y, z-height),(x, y-width, z-height),(x, y-width, z),(x+offs, y-width, z),(x+offs, y-width, z-height),(x+offs, y, z-height),(x+offs, y, z),(x+(2*offs), y-offs, z),(x+(2*offs), y-offs, z-height),(x-offs, y-offs, z-height),(x-offs, y-offs, z),(x-offs, y-(2*offs), z), (x-offs, y-(2*offs), z-height), (x+(2*offs), y-(2*offs), z-height)]

        try:
            # Set the desired orientation for the end effector HERE (marker touches board)
            request.ik_request.pose_stamped.pose.position.x = trans.transform.translation.x
            request.ik_request.pose_stamped.pose.position.y = trans.transform.translation.y
            request.ik_request.pose_stamped.pose.position.z = trans.transform.translation.z - 0.25 # TODO: adjust        

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
            request.ik_request.pose_stamped.pose.position.x += 0.25 # TODO: adjust
            request.ik_request.pose_stamped.pose.position.y -= 0.25 # TODO: adjust

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
            request.ik_request.pose_stamped.pose.position.z += 0.25

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
            request.ik_request.pose_stamped.pose.position.y += 0.25

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
            request.ik_request.pose_stamped.pose.position.z -= 0.25

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
            request.ik_request.pose_stamped.pose.position.x -= 0.25
            request.ik_request.pose_stamped.pose.position.y -= 0.25

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

