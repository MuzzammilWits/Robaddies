#!/usr/bin/env python
import rospy
from navigator import Navigator


if __name__ == "__main__":
    nav = Navigator()

    while nav.map_data is None and not rospy.is_shutdown():
        rospy.sleep(0.5)

    rospy.loginfo("Starting automatic demo route...")

    goals = [
        (0.5, 0.0),
        (0.5, 0.5),
        (0.0, 0.5),
        (0.0, 0.0)
    ]

    for x, y in goals:
        if rospy.is_shutdown():
            break

        rospy.loginfo("Navigating to demo goal: %.2f %.2f", x, y)
        nav.navigate_to(x, y)
        rospy.sleep(1.0)

    rospy.loginfo("Demo route complete.")
