#! /usr/bin/env python
"""
SDK Joint Position Example: keyboard
"""
import argparse

import rospy

import intera_interface
import intera_external_devices

from intera_interface import CHECK_VERSION




def go_to_angles(side, angles):


    limb = intera_interface.Limb(side)
    joints = limb.joint_names()

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
        angle = float(input(f"{joint} Angle:"))
        joint_command[joint] = angle

        delta = 1
        while not rospy.is_shutdown() and abs(delta) > 0.01:
            curr = limb.joint_angle(joint)
            delta = angle - curr
            limb.set_joint_position_speed(0.3)
            limb.set_joint_positions(joint_command)
            rate.sleep()
        
def main():
    """RSDK Joint Position Example: Keyboard Control

    Use your dev machine's keyboard to control joint positions.

    Each key corresponds to increasing or decreasing the angle
    of a joint on Sawyer's arm. The increasing and descreasing
    are represented by number key and letter key next to the number.
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
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__,
                                     epilog=epilog)
    parser.add_argument(
        "-l", "--limb", dest="limb", default=valid_limbs[0],
        choices=valid_limbs,
        help="Limb on which to run the joint position keyboard example"
    )
    args = parser.parse_args(rospy.myargv()[1:])
    angles [0] * 7

    print("Initializing node... ")
    rospy.init_node("sdk_joint_position_keyboard")
    print("Getting robot state... ")
    rs = intera_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nExiting example.")

    rospy.on_shutdown(clean_shutdown)

    rospy.loginfo("Enabling robot...")
    rs.enable()
    # CODE START
    go_to_angles(args.limb, angles)
    # CODE END
    print("Done.")


if __name__ == '__main__':
    main()