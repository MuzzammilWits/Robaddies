#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import sys
import select
import termios
import tty


def publish_velocity(pub, linear_velocity, angular_velocity):
    twist = Twist()
    twist.linear.x = linear_velocity
    twist.angular.z = angular_velocity
    pub.publish(twist)


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


if __name__ == '__main__':
    rospy.init_node('turtlebot_controller', anonymous=True)
    pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=10)

    settings = termios.tcgetattr(sys.stdin)

    linear_velocity = 0.2
    angular_velocity = 0.5

    try:
        while not rospy.is_shutdown():
            key = get_key(settings)

            if key == 'w':
                publish_velocity(pub, linear_velocity, 0.0)
            elif key == 's':
                publish_velocity(pub, -linear_velocity, 0.0)
            elif key == 'a':
                publish_velocity(pub, 0.0, angular_velocity)
            elif key == 'd':
                publish_velocity(pub, 0.0, -angular_velocity)
            elif key == 'q':
                break
            else:
                publish_velocity(pub, 0.0, 0.0)

            rospy.sleep(0.1)

    except rospy.ROSInterruptException:
        pass

    finally:
        publish_velocity(pub, 0.0, 0.0)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
