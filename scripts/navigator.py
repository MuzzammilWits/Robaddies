#!/usr/bin/env python
import rospy
import math
import heapq
import threading

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState


class Navigator:
    def __init__(self):
        rospy.init_node("navigator")

        self.map_data = None
        self.map_width = None
        self.map_height = None
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None

        self.robot_name = rospy.get_param("~robot_name", "mobile_base")

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)

        rospy.wait_for_service("/gazebo/get_model_state")
        self.get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        rospy.loginfo("Navigator ready. Waiting for map...")

    def map_callback(self, msg):
        self.map_data = list(msg.data)
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

    def world_to_grid(self, x, y):
        gx = int((x - self.map_origin_x) / self.map_resolution)
        gy = int((y - self.map_origin_y) / self.map_resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = gx * self.map_resolution + self.map_origin_x
        y = gy * self.map_resolution + self.map_origin_y
        return x, y

    def in_bounds(self, cell):
        x, y = cell
        return 0 <= x < self.map_width and 0 <= y < self.map_height

    def cell_index(self, cell):
        x, y = cell
        return y * self.map_width + x

    def is_free(self, cell):
        if not self.in_bounds(cell):
            return False

        value = self.map_data[self.cell_index(cell)]

        # -1 = unknown, 0 = free, 100 = occupied
        return value == 0

    def inflate_obstacles(self, radius_cells=3):
        inflated = set()

        for y in range(self.map_height):
            for x in range(self.map_width):
                index = y * self.map_width + x
                value = self.map_data[index]

                if value > 50:
                    for dy in range(-radius_cells, radius_cells + 1):
                        for dx in range(-radius_cells, radius_cells + 1):
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                                inflated.add((nx, ny))

        return inflated

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def get_neighbors(self, cell):
        x, y = cell

        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        result = []

        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if self.in_bounds(neighbor):
                result.append(neighbor)

        return result

    def astar(self, start, goal):
        inflated_obstacles = self.inflate_obstacles(radius_cells=3)

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            if current in visited:
                continue

            visited.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in inflated_obstacles:
                    continue

                if not self.is_free(neighbor):
                    continue

                movement_cost = self.heuristic(current, neighbor)
                tentative_g = g_score[current] + movement_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None

    def reconstruct_path(self, came_from, current):
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    def simplify_path(self, path, step=6):
        if len(path) <= 2:
            return path

        simplified = path[::step]

        if simplified[-1] != path[-1]:
            simplified.append(path[-1])

        return simplified

    def get_robot_pose(self):
        try:
            response = self.get_state(self.robot_name, "world")

            x = response.pose.position.x
            y = response.pose.position.y

            q = response.pose.orientation
            yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

            return x, y, yaw

        except rospy.ServiceException as e:
            rospy.logerr("Failed to get robot pose: %s", e)
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
        twist = Twist()
        self.cmd_pub.publish(twist)

    def follow_path(self, world_path):
        rate = rospy.Rate(10)

        k_linear = 0.25
        k_angular = 1.5

        max_linear = 0.25
        max_angular = 0.8

        waypoint_tolerance = 0.20
        final_tolerance = 0.25

        for i, waypoint in enumerate(world_path):
            wx, wy = waypoint

            while not rospy.is_shutdown():
                pose = self.get_robot_pose()

                if pose is None:
                    rate.sleep()
                    continue

                rx, ry, yaw = pose

                dx = wx - rx
                dy = wy - ry

                distance = math.sqrt(dx * dx + dy * dy)

                if distance < waypoint_tolerance:
                    break

                target_angle = math.atan2(dy, dx)
                angle_error = self.normalize_angle(target_angle - yaw)

                twist = Twist()

                if abs(angle_error) > 0.35:
                    twist.linear.x = 0.0
                else:
                    twist.linear.x = min(k_linear * distance, max_linear)

                twist.angular.z = max(
                    min(k_angular * angle_error, max_angular),
                    -max_angular
                )

                self.cmd_pub.publish(twist)
                rate.sleep()

        self.stop_robot()

        pose = self.get_robot_pose()
        if pose is not None:
            rx, ry, _ = pose
            gx, gy = world_path[-1]
            error = math.sqrt((gx - rx) ** 2 + (gy - ry) ** 2)

            if error <= final_tolerance:
                rospy.loginfo("Goal reached.")
            else:
                rospy.logwarn("Stopped, but final error is %.2f m", error)

    def navigate_to(self, goal_x, goal_y):
        if self.map_data is None:
            rospy.logwarn("No map received yet.")
            return

        pose = self.get_robot_pose()

        if pose is None:
            rospy.logerr("Could not get robot pose.")
            return

        start_x, start_y, _ = pose

        start_cell = self.world_to_grid(start_x, start_y)
        goal_cell = self.world_to_grid(goal_x, goal_y)

        rospy.loginfo("Start cell: %s", start_cell)
        rospy.loginfo("Goal cell: %s", goal_cell)

        if not self.in_bounds(goal_cell):
            rospy.logerr("Goal is outside the map.")
            return

        if not self.is_free(goal_cell):
            rospy.logerr("Goal is not in free space.")
            return

        path = self.astar(start_cell, goal_cell)

        if path is None:
            rospy.logerr("No path found.")
            return

        rospy.loginfo("Path found with %d cells.", len(path))

        path = self.simplify_path(path, step=6)
        world_path = [self.grid_to_world(cell[0], cell[1]) for cell in path]

        rospy.loginfo("Following %d waypoints.", len(world_path))
        self.follow_path(world_path)


def input_thread(nav):
    while not rospy.is_shutdown():
        try:
            raw = raw_input("Enter goal x y: ")
            parts = raw.strip().split()

            if len(parts) != 2:
                print("Please enter coordinates like: 2.0 3.5")
                continue

            x = float(parts[0])
            y = float(parts[1])

            nav.navigate_to(x, y)

        except ValueError:
            print("Invalid input. Example: 2.0 3.5")
        except EOFError:
            break


if __name__ == "__main__":
    nav = Navigator()

    while nav.map_data is None and not rospy.is_shutdown():
        rospy.sleep(0.5)

    thread = threading.Thread(target=input_thread, args=(nav,))
    thread.daemon = True
    thread.start()

    rospy.spin()