#!/usr/bin/env python
import os
import sys
import math
import threading

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import GetModelState

# Allow importing sibling script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from path_mapping_clean import plan_path


# TEMPORARY test map path: old project's map on your Desktop
MAP_YAML_PATH = '/mnt/c/Users/whata/Robaddies/maps/surveillance_map.yaml'


class NavigatorClean(object):
    def __init__(self):
        rospy.init_node('navigate_clean')

        self.robot_name = rospy.get_param('~robot_name', 'mobile_base')

        self.front_scan_distance = float('inf')
        self.left_scan_distance = float('inf')
        self.right_scan_distance = float('inf')

        self.cmd_pub = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        rospy.loginfo('navigate_clean ready.')

    def scan_callback(self, msg):
        front_vals = []
        left_vals = []
        right_vals = []

        angle = msg.angle_min
        for r in msg.ranges:
            if not math.isnan(r) and not math.isinf(r) and r > 0.02:
                if abs(angle) <= 0.60:
                    front_vals.append(r)
                elif 0.60 < angle <= 1.57:
                    left_vals.append(r)
                elif -1.57 <= angle < -0.60:
                    right_vals.append(r)
            angle += msg.angle_increment

        self.front_scan_distance = min(front_vals) if front_vals else float('inf')
        self.left_scan_distance = min(left_vals) if left_vals else float('inf')
        self.right_scan_distance = min(right_vals) if right_vals else float('inf')

    def get_robot_pose(self):
        try:
            response = self.get_state(self.robot_name, '')
            x = response.pose.position.x
            y = response.pose.position.y
            q = response.pose.orientation
            yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
            return x, y, yaw
        except rospy.ServiceException as e:
            rospy.logerr('Failed to get gazebo model state: %s', e)
            return None

    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def avoidance_turn(self):
        twist = Twist()
        twist.linear.x = 0.0

        if self.left_scan_distance > self.right_scan_distance:
            twist.angular.z = 0.45
        else:
            twist.angular.z = -0.45

        rate = rospy.Rate(10)
        for _ in range(5):
            if rospy.is_shutdown():
                break
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_robot()

    def move_to_waypoint(self, waypoint):
        rate = rospy.Rate(10)

        max_linear = 0.06
        max_angular = 0.55
        waypoint_tolerance = 0.10

        wx, wy = waypoint

        while not rospy.is_shutdown():
            if self.front_scan_distance < 0.35:
                rospy.logwarn_throttle(1.0, 'Obstacle ahead: %.2f m', self.front_scan_distance)
                self.avoidance_turn()
                return False

            pose = self.get_robot_pose()
            if pose is None:
                rate.sleep()
                continue

            rx, ry, yaw = pose
            dx = wx - rx
            dy = wy - ry
            distance = math.hypot(dx, dy)

            if distance < waypoint_tolerance:
                self.stop_robot()
                return True

            target_angle = math.atan2(dy, dx)
            angle_error = self.normalize_angle(target_angle - yaw)

            twist = Twist()

            # Turn first, then move slowly
            if abs(angle_error) > 0.10:
                twist.linear.x = 0.0
                twist.angular.z = max(min(1.6 * angle_error, max_angular), -max_angular)
            else:
                twist.linear.x = min(max_linear, 0.20 * distance)
                if self.front_scan_distance < 0.50:
                    twist.linear.x = min(twist.linear.x, 0.02)
                twist.angular.z = max(min(0.8 * angle_error, max_angular), -max_angular)

            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_robot()
        return False

    def navigate_to(self, goal):
        goal_tolerance = 0.15
        max_replans = 20
        short_horizon_waypoints = 4

        for attempt in range(max_replans):
            pose = self.get_robot_pose()
            if pose is None:
                rospy.logerr('Could not get current pose.')
                return

            start = pose[:2]

            if math.hypot(goal[0] - start[0], goal[1] - start[1]) < goal_tolerance:
                self.stop_robot()
                rospy.loginfo('Goal reached.')
                return

            rospy.loginfo('Planning path from %s to %s (replan %d/%d)...',
                          str((round(start[0], 2), round(start[1], 2))),
                          str((round(goal[0], 2), round(goal[1], 2))),
                          attempt + 1, max_replans)

            try:
                path = plan_path(
                    start,
                    goal,
                    map_yaml_path=MAP_YAML_PATH,
                    inflation_radius_m=0.18,
                    grid_step_m=0.75,
                    max_connection_distance_m=2.5,
                    debug_plot=True
                )
            except Exception as e:
                rospy.logerr('Planner failed: %s', e)
                return

            if not path or len(path) < 2:
                rospy.logerr('No usable path found.')
                return

            # Drop path points that are basically on top of current pose
            cleaned = []
            for p in path:
                if math.hypot(p[0] - start[0], p[1] - start[1]) > 0.12:
                    cleaned.append(p)

            if not cleaned:
                self.stop_robot()
                rospy.loginfo('Goal reached.')
                return

            segment = cleaned[:short_horizon_waypoints]

            rospy.loginfo('Planner returned %d waypoints, following next %d.',
                          len(cleaned), len(segment))

            interrupted = False
            for waypoint in segment:
                ok = self.move_to_waypoint(waypoint)
                if not ok:
                    interrupted = True
                    break

                pose = self.get_robot_pose()
                if pose is not None:
                    if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) < goal_tolerance:
                        self.stop_robot()
                        rospy.loginfo('Goal reached.')
                        return

            if interrupted:
                rospy.logwarn('Segment interrupted, replanning...')
                continue

        self.stop_robot()
        rospy.logwarn('Reached replan limit without finishing.')


def input_thread(nav):
    while not rospy.is_shutdown():
        try:
            raw = raw_input('Enter goal x y: ').strip()
            parts = raw.split()

            if len(parts) != 2:
                print('Please enter coordinates like: 2.0 3.5')
                continue

            goal_x = float(parts[0])
            goal_y = float(parts[1])
            nav.navigate_to((goal_x, goal_y))

        except ValueError:
            print('Invalid input. Example: 2.0 3.5')
        except EOFError:
            break


if __name__ == '__main__':
    nav = NavigatorClean()
    thread = threading.Thread(target=input_thread, args=(nav,))
    thread.daemon = True
    thread.start()
    rospy.spin()
