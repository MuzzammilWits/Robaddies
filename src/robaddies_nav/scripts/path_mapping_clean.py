#!/usr/bin/env python
import os
import math
import heapq

import cv2
import yaml
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class RoadmapPlanner(object):
    def __init__(self,
                 map_yaml_path=None,
                 inflation_radius_m=0.18,
                 grid_step_m=0.75,
                 max_connection_distance_m=2.5,
                 debug_plot=True):
        self.map_yaml_path = map_yaml_path or self._default_map_yaml_path()
        self.inflation_radius_m = inflation_radius_m
        self.grid_step_m = grid_step_m
        self.max_connection_distance_m = max_connection_distance_m
        self.debug_plot = debug_plot

        self.map_image = None
        self.free_mask = None
        self.blocked_mask = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.height = None
        self.width = None
        self.image_path = None

        self._load_map()

    def _default_map_yaml_path(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(script_dir)   # .../src/robaddies_nav
        src_dir = os.path.dirname(package_dir)      # .../src
        repo_root = os.path.dirname(src_dir)        # .../Robaddies
        return os.path.join(repo_root, 'maps', 'surveillance_map.yaml')

    def _load_map(self):
        if not os.path.exists(self.map_yaml_path):
            raise IOError("Map yaml not found: %s" % self.map_yaml_path)

        with open(self.map_yaml_path, 'r') as f:
            meta = yaml.safe_load(f)

        self.resolution = float(meta['resolution'])
        self.origin_x = float(meta['origin'][0])
        self.origin_y = float(meta['origin'][1])

        image_name = meta['image']
        if os.path.isabs(image_name):
            self.image_path = image_name
        else:
            self.image_path = os.path.join(os.path.dirname(self.map_yaml_path), image_name)

        self.map_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.map_image is None:
            raise IOError("Failed to load map image: %s" % self.image_path)

        self.height, self.width = self.map_image.shape[:2]

        occupied_thresh = float(meta.get('occupied_thresh', 0.65))
        free_thresh = float(meta.get('free_thresh', 0.196))
        negate = int(meta.get('negate', 0))

        image = self.map_image.astype(np.float32) / 255.0
        if negate == 0:
            occ = 1.0 - image
        else:
            occ = image

        occupied_mask = occ > occupied_thresh
        free_mask = occ < free_thresh
        unknown_mask = np.logical_not(np.logical_or(occupied_mask, free_mask))

        inflation_radius_px = int(math.ceil(self.inflation_radius_m / self.resolution))
        kernel_size = 2 * inflation_radius_px + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)

        for r in range(kernel_size):
            for c in range(kernel_size):
                dr = r - inflation_radius_px
                dc = c - inflation_radius_px
                if dr * dr + dc * dc <= inflation_radius_px * inflation_radius_px:
                    kernel[r, c] = 1

        occupied_u8 = occupied_mask.astype(np.uint8)
        inflated_occupied = cv2.dilate(occupied_u8, kernel, iterations=1).astype(bool)

        self.blocked_mask = np.logical_or(inflated_occupied, unknown_mask)
        self.free_mask = np.logical_not(self.blocked_mask)

    def world_to_pixel(self, point):
        x, y = point
        col = int((x - self.origin_x) / self.resolution)
        map_y = int((y - self.origin_y) / self.resolution)
        row = self.height - 1 - map_y
        return (row, col)

    def pixel_to_world(self, pixel):
        row, col = pixel
        map_y = self.height - 1 - row
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (map_y + 0.5) * self.resolution
        return (x, y)

    def in_bounds_pixel(self, pixel):
        row, col = pixel
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free_pixel(self, pixel):
        if not self.in_bounds_pixel(pixel):
            return False
        row, col = pixel
        return bool(self.free_mask[row, col])

    def nearest_free_pixel(self, pixel, max_radius_px=80):
        if self.is_free_pixel(pixel):
            return pixel

        row0, col0 = pixel
        best = None
        best_dist = None

        for radius in range(1, max_radius_px + 1):
            rmin = row0 - radius
            rmax = row0 + radius
            cmin = col0 - radius
            cmax = col0 + radius

            for col in range(cmin, cmax + 1):
                for row in (rmin, rmax):
                    p = (row, col)
                    if self.is_free_pixel(p):
                        d = abs(row - row0) + abs(col - col0)
                        if best is None or d < best_dist:
                            best = p
                            best_dist = d

            for row in range(rmin + 1, rmax):
                for col in (cmin, cmax):
                    p = (row, col)
                    if self.is_free_pixel(p):
                        d = abs(row - row0) + abs(col - col0)
                        if best is None or d < best_dist:
                            best = p
                            best_dist = d

            if best is not None:
                return best

        return None

    def raster_line(self, p0, p1):
        r0, c0 = p0
        r1, c1 = p1

        points = []

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)

        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1

        err = dc - dr

        r = r0
        c = c0

        while True:
            points.append((r, c))
            if r == r1 and c == c1:
                break

            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr

        return points

    def is_visible(self, p0, p1):
        line = self.raster_line(p0, p1)
        for pixel in line:
            if not self.is_free_pixel(pixel):
                return False
        return True

    def sample_nodes(self, start_pixel, goal_pixel):
        step_px = int(max(6, round(self.grid_step_m / self.resolution)))
        nodes = []

        row_start = step_px // 2
        col_start = step_px // 2

        for row in range(row_start, self.height, step_px):
            for col in range(col_start, self.width, step_px):
                p = (row, col)
                if self.is_free_pixel(p):
                    nodes.append(p)

        nodes.append(start_pixel)
        nodes.append(goal_pixel)

        seen = set()
        unique = []
        for p in nodes:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        return unique

    def build_graph(self, nodes):
        adjacency = {}
        for i in range(len(nodes)):
            adjacency[i] = []

        max_dist_px = self.max_connection_distance_m / self.resolution

        for i in range(len(nodes)):
            r1, c1 = nodes[i]
            for j in range(i + 1, len(nodes)):
                r2, c2 = nodes[j]

                dr = r2 - r1
                dc = c2 - c1
                dist_px = math.hypot(dr, dc)

                if dist_px > max_dist_px:
                    continue

                if self.is_visible(nodes[i], nodes[j]):
                    dist_m = dist_px * self.resolution
                    adjacency[i].append((j, dist_m))
                    adjacency[j].append((i, dist_m))

        return adjacency

    def dijkstra(self, adjacency, start_idx, goal_idx):
        pq = [(0.0, start_idx)]
        dist = {start_idx: 0.0}
        prev = {start_idx: None}
        visited = set()

        while pq:
            current_cost, u = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            if u == goal_idx:
                break

            for v, w in adjacency[u]:
                new_cost = current_cost + w
                if v not in dist or new_cost < dist[v]:
                    dist[v] = new_cost
                    prev[v] = u
                    heapq.heappush(pq, (new_cost, v))

        if goal_idx not in prev:
            return None

        path_indices = []
        node = goal_idx
        while node is not None:
            path_indices.append(node)
            node = prev[node]
        path_indices.reverse()
        return path_indices

    def save_debug_plot(self, nodes, adjacency, path_indices, output_path=None):
        if not self.debug_plot:
            return

        if output_path is None:
            output_path = os.path.join(os.getcwd(), 'planner_debug_plot.png')

        obstacle_pixels = np.argwhere(self.blocked_mask)
        obstacle_world = np.array([self.pixel_to_world((int(r), int(c))) for r, c in obstacle_pixels])

        node_world = np.array([self.pixel_to_world(p) for p in nodes])

        plt.figure(figsize=(10, 10))

        if len(obstacle_world) > 0:
            plt.scatter(obstacle_world[:, 0], obstacle_world[:, 1], c='b', s=1, label='obstacles')

        if len(node_world) > 0:
            plt.scatter(node_world[:, 0], node_world[:, 1], c='r', s=12, label='nodes')

        for i in adjacency:
            for j, _ in adjacency[i]:
                if j <= i:
                    continue
                p1 = self.pixel_to_world(nodes[i])
                p2 = self.pixel_to_world(nodes[j])
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=0.4)

        if path_indices is not None and len(path_indices) >= 2:
            path_points = np.array([self.pixel_to_world(nodes[idx]) for idx in path_indices])
            plt.plot(path_points[:, 0], path_points[:, 1], 'm-', linewidth=2.5, label='path')

        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Roadmap planner debug')
        plt.axis('equal')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plan_path(self, start_world, goal_world):
        start_pixel_raw = self.world_to_pixel(start_world)
        goal_pixel_raw = self.world_to_pixel(goal_world)

        start_pixel = self.nearest_free_pixel(start_pixel_raw)
        goal_pixel = self.nearest_free_pixel(goal_pixel_raw)

        if start_pixel is None:
            raise RuntimeError("Could not find free start pixel near %s" % (start_world,))
        if goal_pixel is None:
            raise RuntimeError("Could not find free goal pixel near %s" % (goal_world,))

        nodes = self.sample_nodes(start_pixel, goal_pixel)
        start_idx = len(nodes) - 2
        goal_idx = len(nodes) - 1

        adjacency = self.build_graph(nodes)
        path_indices = self.dijkstra(adjacency, start_idx, goal_idx)

        self.save_debug_plot(nodes, adjacency, path_indices)

        if path_indices is None:
            return []

        world_path = [self.pixel_to_world(nodes[idx]) for idx in path_indices]
        return world_path


def plan_path(start_pos, end_pos,
              map_yaml_path=None,
              inflation_radius_m=0.18,
              grid_step_m=0.75,
              max_connection_distance_m=2.5,
              debug_plot=True):
        planner = RoadmapPlanner(
            map_yaml_path=map_yaml_path,
            inflation_radius_m=inflation_radius_m,
            grid_step_m=grid_step_m,
            max_connection_distance_m=max_connection_distance_m,
            debug_plot=debug_plot
        )
        return planner.plan_path(start_pos, end_pos)


if __name__ == '__main__':
    start = (0.0, 0.0)
    goal = (1.0, 1.0)
    path = plan_path(start, goal)
    print(path)
