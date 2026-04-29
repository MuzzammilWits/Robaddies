#!/usr/bin/env python
import rospy, math, heapq, threading
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState

class Navigator:
    def __init__(self):
        rospy.init_node('navigator')
        self.map_data = None
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0.05
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.robot_name = rospy.get_param('~robot_name', 'mobile_base')
        self.cmd_pub = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        rospy.loginfo('Navigator ready. Waiting for map...')

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
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        x = gx * self.map_resolution + self.map_origin_x
        y = gy * self.map_resolution + self.map_origin_y
        return (x, y)

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
        return value >= 0 and value < 50

    def heuristic(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def neighbors(self, cell):
        x, y = cell
        dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        out = []
        for dx, dy in dirs:
            n = (x+dx, y+dy)
            if self.in_bounds(n) and self.is_free(n):
                out.append(n)
        return out

    def astar(self, start, goal):
        open_set = [(0, start)]
        came = {}
        g = {start: 0.0}
        seen = set()
        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return list(reversed(path))
            if cur in seen:
                continue
            seen.add(cur)
            for n in self.neighbors(cur):
                ng = g[cur] + self.heuristic(cur, n)
                if n not in g or ng < g[n]:
                    g[n] = ng
                    came[n] = cur
                    heapq.heappush(open_set, (ng + self.heuristic(n, goal), n))
        return None

    def yaw_from_q(self, q):
        siny = 2.0 * (q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny, cosy)

    def pose(self):
        r = self.get_state(self.robot_name, 'world')
        return r.pose.position.x, r.pose.position.y, self.yaw_from_q(r.pose.orientation)

    def norm(self, a):
        while a > math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def follow(self, path):
        rate = rospy.Rate(10)
        for gx, gy in path:
            wx, wy = self.grid_to_world(gx, gy)
            while not rospy.is_shutdown():
                rx, ry, yaw = self.pose()
                dx, dy = wx-rx, wy-ry
                dist = math.hypot(dx, dy)
                if dist < 0.18:
                    break
                ang = math.atan2(dy, dx)
                err = self.norm(ang - yaw)
                t = Twist()
                t.angular.z = max(min(1.2*err, 0.8), -0.8)
                if abs(err) < 0.4:
                    t.linear.x = min(0.25, 0.25*dist)
                self.cmd_pub.publish(t)
                rate.sleep()
        self.cmd_pub.publish(Twist())

    def navigate_to(self, x, y):
        if self.map_data is None:
            rospy.logwarn('No map yet')
            return
        sx, sy, _ = self.pose()
        start = self.world_to_grid(sx, sy)
        goal = self.world_to_grid(x, y)
        rospy.loginfo('Start cell: %s', start)
        rospy.loginfo('Goal cell: %s', goal)
        if not self.is_free(goal):
            rospy.logerr('Goal is not in free space.')
            return
        path = self.astar(start, goal)
        if not path:
            rospy.logerr('No path found.')
            return
        rospy.loginfo('Path found with %d cells.', len(path))
        self.follow(path[::6] + [path[-1]])


def input_thread(nav):
    while not rospy.is_shutdown():
        try:
            raw = raw_input('Enter goal x y: ')
            p = raw.strip().split()
            if len(p) != 2:
                continue
            nav.navigate_to(float(p[0]), float(p[1]))
        except Exception:
            pass

if __name__ == '__main__':
    nav = Navigator()
    while nav.map_data is None and not rospy.is_shutdown():
        rospy.sleep(0.5)
    th = threading.Thread(target=input_thread, args=(nav,))
    th.daemon = True
    th.start()
    rospy.spin()

