#!/usr/bin/env python
import rospy
import tf2_ros
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

    while not rospy.is_shutdown():
        input('Press [ Enter ]: ')
        
        # Construct the request
        request = GetPositionIKRequest()
        request.ik_request.group_name = "right_arm"

        # If a Sawyer does not have a gripper, replace '_gripper_tip' with '_wrist' instead
        link = "stp_022310TP99251_tip" #"right_gripper_tip"

        request.ik_request.ik_link_name = link
        # request.ik_request.attempts = 20
        request.ik_request.pose_stamped.header.frame_id = "base"
        request.ik_request.pose_stamped.pose.orientation.x = 0
        request.ik_request.pose_stamped.pose.orientation.y = 1
        request.ik_request.pose_stamped.pose.orientation.z = 0
        request.ik_request.pose_stamped.pose.orientation.w = 0

        width = 0.2
        offs = width/3
        height = 0.02
        
        try:
            trans = tfBuffer.lookup_transform("base", "stp_022310TP99251_tip", rospy.Time())   # TODO: may need to update frames
            # trans.transform.translation gives current x, y, z
            
            # find curr location of end effector:
            x = trans.transform.translation.x   # TODO: grid loc x
            y = trans.transform.translation.y   # TODO: grid loc y
            z = 0.016 #trans.transform.translation.z   # TODO: grid loc z (marker lifted)
            
            # trans.transform.translation gives current x, y, z
            locs = [(x, y, z), (x, y-width/3, z), (x, y-2*width/3, z), (x, y-width, z),(x, y-width, z+height),(x+offs, y-width, z+height), # border horizontal
                    (x+offs, y-width, z), (x+offs, y-2*width/3, z), (x+offs, y-width/3, z), (x+offs, y, z),(x+offs, y, z+height),(x+(2*offs), y, z+height), #horizontal 1
                    (x+(2*offs), y, z), (x+(2*offs), y-width/3, z), (x+(2*offs), y-2*width/3, z), (x+(2*offs), y-width, z),(x+(2*offs), y-width, z+height),(x+width, y-width, z+height), #horizontal 2
                    (x+width, y-width, z), (x+width, y-2*width/3, z), (x+width, y-width/3, z), (x+width, y, z), (x+width, y, z+height) #(x+width, y, z+height),(x+width, y-offs, z+height), # border horizontal


                    # (x+(2*offs), y, z), (x+(offs), y, z), (x, y, z),(x, y, z+height),(x, y-offs, z+height),  # verical border
                    # (x, y-offs, z), (x+offs, y-offs, z), (x+ (2 *offs), y-offs, z),(x+width, y-offs, z), (x+width, y-offs, z+height),(x+width, y-(2*offs), z+height), # first vertical
                    # (x+width, y-(2*offs), z), (x+(2 * offs), y-(2*offs), z), (x + offs, y-(2*offs), z), (x, y-(2*offs), z), (x, y-(2*offs), z+height), (x, y-width, z+height), #second vertical
                    # (x, y-width, z), (x+offs, y-width, z), (x+ (2 *offs), y-width, z),(x+width, y-width, z) #(x+width, y-offs, z+height),(x+width, y-(2*offs), z+height) # verical border
                    
                    ]
            
            # new_locs = [[x1, y1, z1, 0, 1, 0, 0] for x1, y1, z1 in locs]
            # response = compute_ik(request)
                    
            # # Print the response HERE
            # print(response)
            # group = MoveGroupCommander("right_arm")

            # # Setting position and orientation target
            # group.set_pose_targets(new_locs)

            # # Plan IK
            # plan = group.plan()
            # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # # # Execute IK if safe
            # # # if user_input == 'y':
            # # #     group.execute(plan[1])

            # # while user_input == 'n':
            # #     plan = group.plan()
            # #     user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")

            
            # if user_input == 'y':  
            #     group.execute(plan[1])
            
            # for i in range(2):
            for x1, y1, z1 in locs:
                    
                    # Set the desired orientation for the end effector HERE (marker touches board)
                request.ik_request.pose_stamped.pose.position.x = x1
                request.ik_request.pose_stamped.pose.position.y = y1
                request.ik_request.pose_stamped.pose.position.z = z1 # TODO: adjust        

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
                # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
                
                # # Execute IK if safe
                # # if user_input == 'y':
                # #     group.execute(plan[1])

                # while user_input == 'n':
                #     plan = group.plan()
                #     user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")

                
                # if user_input == 'y':
                group.execute(plan[1])  


            for x1, y1, z1 in [(x+width, y, z+height), (x+width, y, z+height)] + locs[::-1] + [(0.694, 0.158, 0.525)]: # last coord is tuck
                    
                # Set the desired orientation for the end effector HERE (marker touches board)
                request.ik_request.pose_stamped.pose.position.x = x1
                request.ik_request.pose_stamped.pose.position.y = y1
                request.ik_request.pose_stamped.pose.position.z = z1 # TODO: adjust        

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
                # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
                
                # # Execute IK if safe
                # # if user_input == 'y':
                # #     group.execute(plan[1])

                # while user_input == 'n':
                #     plan = group.plan()
                #     user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")

                
                # if user_input == 'y':
                group.execute(plan[1])  
            # ---------------------------------------------
            # 
            # 
            #     
            # ---------------------------------------------  

            # # move robot to loc 2 (draw first line of X)
            # request.ik_request.pose_stamped.pose.position.x += 0.25 # TODO: adjust
            # request.ik_request.pose_stamped.pose.position.y -= 0.25 # TODO: adjust

            # # Send the request to the service
            # response = compute_ik(request)
            
            # # Print the response HERE
            # print(response)
            # group = MoveGroupCommander("right_arm")

            # # Setting position and orientation target
            # group.set_pose_target(request.ik_request.pose_stamped)

            # # Plan IK
            # plan = group.plan()
            # # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # # Execute IK if safe
            # # if user_input == 'y':
            # group.execute(plan[1])

            # # move robot to loc 3 (lift marker)
            # request.ik_request.pose_stamped.pose.position.z += 0.25

            # # Send the request to the service
            # response = compute_ik(request)
            
            # # Print the response HERE
            # print(response)
            # group = MoveGroupCommander("right_arm")

            # # Setting position and orientation target
            # group.set_pose_target(request.ik_request.pose_stamped)

            # # Plan IK
            # plan = group.plan()
            # # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # # Execute IK if safe
            # # if user_input == 'y':
            # group.execute(plan[1])

            #  # move robot to loc 4 (location of second line)
            # request.ik_request.pose_stamped.pose.position.y += 0.25

            # # Send the request to the service
            # response = compute_ik(request)
            
            # # Print the response HERE
            # print(response)
            # group = MoveGroupCommander("right_arm")

            # # Setting position and orientation target
            # group.set_pose_target(request.ik_request.pose_stamped)

            # # Plan IK
            # plan = group.plan()
            # # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # # Execute IK if safe
            # # if user_input == 'y':
            # group.execute(plan[1])

            #  # move robot to loc 5 (marker touches board)
            # request.ik_request.pose_stamped.pose.position.z -= 0.25

            # # Send the request to the service
            # response = compute_ik(request)
            
            # # Print the response HERE
            # print(response)
            # group = MoveGroupCommander("right_arm")

            # # Setting position and orientation target
            # group.set_pose_target(request.ik_request.pose_stamped)

            # # Plan IK
            # plan = group.plan()
            # # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # # Execute IK if safe
            # # if user_input == 'y':
            # group.execute(plan[1])

            #  # move robot to loc 6 (draw second line of X)
            # request.ik_request.pose_stamped.pose.position.x -= 0.25
            # request.ik_request.pose_stamped.pose.position.y -= 0.25

            # # Send the request to the service
            # response = compute_ik(request)
            
            # # Print the response HERE
            # print(response)
            # group = MoveGroupCommander("right_arm")

            # # Setting position and orientation target
            # group.set_pose_target(request.ik_request.pose_stamped)

            # # Plan IK
            # plan = group.plan()
            # # user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
            
            # # Execute IK if safe
            # # if user_input == 'y':
            # group.execute(plan[1])

            # TODO: move robot back to tuck???
            
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)


# Python's syntax for a main() method
if __name__ == '__main__':
    main()

