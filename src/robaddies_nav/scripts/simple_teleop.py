#!/usr/bin/env python
import rospy
import sys
import termios
import tty
from geometry_msgs.msg import Twist

settings = termios.tcgetattr(sys.stdin)

def get_key():
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

rospy.init_node("simple_teleop")
pub = rospy.Publisher("/cmd_vel_mux/input/navi", Twist, queue_size=10)

print("WASD controls:")
print("w = forward")
print("a = turn left")
print("d = turn right")
print("s = stop")
print("q = quit")

while not rospy.is_shutdown():
    key = get_key()
    t = Twist()

    if key == "w":
        t.linear.x = 0.15
    elif key == "a":
        t.angular.z = 0.5
    elif key == "d":
        t.angular.z = -0.5
    elif key == "s":
        pass
    elif key == "q":
        break

    pub.publish(t)

pub.publish(Twist())
