#! /usr/bin/env python
"""
SDK Joint Position Example: keyboard
"""
import argparse

import rospy

import intera_interface
import intera_external_devices

from intera_interface import CHECK_VERSION


CUSTOM_TUCK = (0.611, 0.183, -0.093) # position

def go_to_angles(side, angles):


    limb = intera_interface.Limb(side)
    joints = limb.joint_names()
    print('joints', joints)

    if len(angles) != len(joints):
        rospy.logerr("Number of provided angles does not match the number of joints.")
        return

    joint_command = {joint: angle for joint, angle in zip(joints, angles)}
    

    rate = rospy.Rate(100)

    
    c = intera_external_devices.getch()
    if c:
        #catch Esc or ctrl-c
        if c in ['\x1b', '\x03']:
            done = True
            rospy.signal_shutdown("Example finished.")

    for joint in joints:
        # angle = float(input(f"{joint} Angle:"))
        # joint_command[joint] = angle
        angle = joint_command[joint]

        delta = 1
        while not rospy.is_shutdown() and abs(delta) > 0.01:
            curr = limb.joint_angle(joint)
            delta = angle - curr
            limb.set_joint_position_speed(0.3)
            limb.set_joint_positions(joint_command)
            rate.sleep()
        
def main():
    """
    Will move to our custom tuck (allows us to draw vertical lines and avoid singualariteis)
    """
    epilog = """
See help inside the example with the '?' key for key bindings.
    """
    rp = intera_interface.RobotParams()
    valid_limbs = rp.get_limb_names()
    if not valid_limbs:
        rp.log_message(("Cannot detect any limb parameters on this robot. "
                        "Exiting."), "ERROR")
        return

    # CUSTOM TUCK 
    angles = [0.18581640625, 1.0606513671875, 0.165806640625, -1.8724892578125, -0.1731123046875, 2.20851953125, 1.64812890625]

    # angles = [0.0018544921875, -0.3010791015625, 0.98905078125, -0.523552734375, -1.3970185546875, 0.3013359375, 1.9259921875, 2.126453125, 0.0]
    # angles = angles[1:-1]
    print("Initializing node... ")
    # rospy.init_node("sdk_joint_position_keyboard")
    print("Getting robot state... ")
    rs = intera_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nExiting example.")

    rospy.on_shutdown(clean_shutdown)

    rospy.loginfo("Enabling robot...")
    rs.enable()
    # CODE START
    go_to_angles(valid_limbs[0], angles)
    # CODE END
    print("Done.")


if __name__ == '__main__':
    main()