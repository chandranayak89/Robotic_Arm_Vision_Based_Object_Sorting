"""
Path planning module for robotic arm trajectory generation.
Implements various planning algorithms including RRT, RRT*, and PRM.
"""
import numpy as np
import math
import random
import time
import sys
import os
from enum import Enum
from collections import defaultdict, deque

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PlanningMethod(Enum):
    """Enumeration of supported path planning methods."""
    RRT = "rrt"              # Rapidly-exploring Random Tree
    RRT_STAR = "rrt_star"    # RRT with optimality improvements
    PRM = "prm"              # Probabilistic Roadmap


class Node:
    """Node class for tree-based planning algorithms."""
    
    def __init__(self, position):
        """
        Initialize a node.
        
        Args:
            position: Position as (x, y, z) coordinates
        """
        self.position = position
        self.parent = None
        self.cost = 0.0  # Cost from start (for RRT*)
        self.children = []
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return np.array_equal(self.position, other.position)
        return False
    
    def distance_to(self, other_node):
        """Calculate Euclidean distance to another node."""
        return np.linalg.norm(np.array(self.position) - np.array(other_node.position))


class Obstacle:
    """Base class for obstacle representation."""
    
    def __init__(self, position, safety_margin=0):
        """
        Initialize an obstacle.
        
        Args:
            position: Position as (x, y, z) coordinates
            safety_margin: Safety margin to add around the obstacle in mm
        """
        self.position = position
        self.safety_margin = safety_margin
    
    def is_in_collision(self, position):
        """
        Check if a position collides with the obstacle.
        To be implemented by derived classes.
        """
        raise NotImplementedError("Subclasses must implement is_in_collision")


class SphereObstacle(Obstacle):
    """Spherical obstacle representation."""
    
    def __init__(self, position, radius, safety_margin=0):
        """
        Initialize a spherical obstacle.
        
        Args:
            position: Center position as (x, y, z) coordinates
            radius: Radius of the sphere in mm
            safety_margin: Additional safety margin in mm
        """
        super().__init__(position, safety_margin)
        self.radius = radius
    
    def is_in_collision(self, position):
        """
        Check if a position collides with the spherical obstacle.
        
        Args:
            position: Position to check as (x, y, z) coordinates
            
        Returns:
            True if in collision, False otherwise
        """
        distance = np.linalg.norm(np.array(position) - np.array(self.position))
        return distance < (self.radius + self.safety_margin)


class BoxObstacle(Obstacle):
    """Box-shaped obstacle representation."""
    
    def __init__(self, position, dimensions, orientation=(0, 0, 0), safety_margin=0):
        """
        Initialize a box-shaped obstacle.
        
        Args:
            position: Center position as (x, y, z) coordinates
            dimensions: Box dimensions as (width, height, depth) in mm
            orientation: Box orientation as (roll, pitch, yaw) in degrees
            safety_margin: Additional safety margin in mm
        """
        super().__init__(position, safety_margin)
        self.dimensions = dimensions
        self.orientation = orientation
        
        # For simplicity, we'll use an axis-aligned bounding box (AABB) approach
        # For full 3D oriented bounding boxes, a more complex algorithm would be needed
        half_width = dimensions[0] / 2 + safety_margin
        half_height = dimensions[1] / 2 + safety_margin
        half_depth = dimensions[2] / 2 + safety_margin
        
        self.min_bounds = (
            position[0] - half_width,
            position[1] - half_height,
            position[2] - half_depth
        )
        
        self.max_bounds = (
            position[0] + half_width,
            position[1] + half_height,
            position[2] + half_depth
        )
    
    def is_in_collision(self, position):
        """
        Check if a position collides with the box obstacle.
        
        Args:
            position: Position to check as (x, y, z) coordinates
            
        Returns:
            True if in collision, False otherwise
        """
        # Check if the position is within the AABB
        return (position[0] >= self.min_bounds[0] and position[0] <= self.max_bounds[0] and
                position[1] >= self.min_bounds[1] and position[1] <= self.max_bounds[1] and
                position[2] >= self.min_bounds[2] and position[2] <= self.max_bounds[2])


class PathPlanner:
    """Path planner class for robotic arm trajectory planning."""
    
    def __init__(self, method=None, workspace_bounds=None, obstacles=None):
        """
        Initialize the path planner.
        
        Args:
            method: Planning method from PlanningMethod enum
            workspace_bounds: Dictionary with min/max bounds for workspace
            obstacles: List of obstacles in the workspace
        """
        # Use defaults from config if not provided
        self.method = method or PlanningMethod(config.PATH_PLANNING_METHOD)
        
        # Set workspace bounds
        if workspace_bounds is None:
            # Default workspace bounds from config
            self.workspace_bounds = {
                'x_min': 0,
                'x_max': config.WORKSPACE_WIDTH,
                'y_min': 0,
                'y_max': config.WORKSPACE_HEIGHT,
                'z_min': 0,
                'z_max': config.WORKSPACE_DEPTH
            }
        else:
            self.workspace_bounds = workspace_bounds
        
        # Initialize obstacle list
        self.obstacles = obstacles or []
        
        # Planning parameters
        self.step_size = 20.0  # Step size for extending tree in mm
        self.max_iterations = 5000  # Maximum iterations for planning
        self.goal_sample_rate = 0.1  # Probability of sampling the goal
        self.search_radius = 50.0  # Search radius for RRT* and PRM
        
        # For PRM
        self.prm_num_samples = 500  # Number of samples for PRM
        self.prm_max_neighbors = 10  # Maximum number of neighbors for each node
        self.roadmap = None  # Roadmap for PRM

    def add_obstacle(self, obstacle):
        """
        Add an obstacle to the workspace.
        
        Args:
            obstacle: Obstacle object to add
        """
        self.obstacles.append(obstacle)
    
    def clear_obstacles(self):
        """Remove all obstacles from the workspace."""
        self.obstacles = []
    
    def is_collision_free(self, position):
        """
        Check if a position is collision-free.
        
        Args:
            position: Position to check as (x, y, z) coordinates
            
        Returns:
            True if collision-free, False otherwise
        """
        # Check if position is within workspace bounds
        if (position[0] < self.workspace_bounds['x_min'] or
            position[0] > self.workspace_bounds['x_max'] or
            position[1] < self.workspace_bounds['y_min'] or
            position[1] > self.workspace_bounds['y_max'] or
            position[2] < self.workspace_bounds['z_min'] or
            position[2] > self.workspace_bounds['z_max']):
            return False
        
        # Check collision with obstacles
        if config.OBSTACLE_AVOIDANCE_ENABLED:
            for obstacle in self.obstacles:
                if obstacle.is_in_collision(position):
                    return False
        
        return True
    
    def is_path_collision_free(self, start_pos, end_pos, resolution=0.01):
        """
        Check if a path between two positions is collision-free.
        
        Args:
            start_pos: Start position as (x, y, z) coordinates
            end_pos: End position as (x, y, z) coordinates
            resolution: Resolution for interpolation (0-1)
            
        Returns:
            True if collision-free, False otherwise
        """
        # Convert to numpy arrays
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        
        # Calculate distance
        distance = np.linalg.norm(end_pos - start_pos)
        
        # Determine number of interpolation steps
        steps = max(2, int(distance * resolution))
        
        # Interpolate and check collisions
        for i in range(steps + 1):
            t = i / steps
            interp_pos = start_pos * (1 - t) + end_pos * t
            if not self.is_collision_free(interp_pos):
                return False
        
        return True
    
    def sample_random_position(self):
        """
        Sample a random position within the workspace bounds.
        
        Returns:
            Random position as (x, y, z) coordinates
        """
        x = random.uniform(self.workspace_bounds['x_min'], self.workspace_bounds['x_max'])
        y = random.uniform(self.workspace_bounds['y_min'], self.workspace_bounds['y_max'])
        z = random.uniform(self.workspace_bounds['z_min'], self.workspace_bounds['z_max'])
        
        return (x, y, z)

    def find_nearest_node(self, nodes, position):
        """
        Find the nearest node in a list to a given position.
        
        Args:
            nodes: List of nodes
            position: Target position as (x, y, z) coordinates
            
        Returns:
            Nearest node and its index
        """
        distances = [np.linalg.norm(np.array(node.position) - np.array(position)) for node in nodes]
        nearest_idx = np.argmin(distances)
        return nodes[nearest_idx], nearest_idx
    
    def steer(self, from_pos, to_pos, step_size=None):
        """
        Steer from one position toward another, limited by step size.
        
        Args:
            from_pos: Starting position as (x, y, z) coordinates
            to_pos: Target position as (x, y, z) coordinates
            step_size: Maximum step size (defaults to self.step_size)
            
        Returns:
            New position after steering
        """
        if step_size is None:
            step_size = self.step_size
        
        # Convert to numpy arrays
        from_pos = np.array(from_pos)
        to_pos = np.array(to_pos)
        
        # Calculate direction and distance
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        
        # If distance is less than step size, just return target position
        if distance < step_size:
            return tuple(to_pos)
        
        # Otherwise, move in the direction of target by step_size
        normalized_direction = direction / distance
        new_pos = from_pos + normalized_direction * step_size
        
        return tuple(new_pos)
    
    def plan_path_rrt(self, start_pos, goal_pos):
        """
        Plan a path using the Rapidly-exploring Random Tree (RRT) algorithm.
        
        Args:
            start_pos: Start position as (x, y, z) coordinates
            goal_pos: Goal position as (x, y, z) coordinates
            
        Returns:
            List of positions defining the path, or None if no path found
        """
        # Initialize the tree with the start node
        start_node = Node(start_pos)
        goal_node = Node(goal_pos)
        nodes = [start_node]
        
        # Check if direct path is possible
        if self.is_path_collision_free(start_pos, goal_pos):
            goal_node.parent = start_node
            return self._extract_path(goal_node)
        
        # Main RRT loop
        for i in range(self.max_iterations):
            # Sample random position (with bias toward goal)
            if random.random() < self.goal_sample_rate:
                random_pos = goal_pos
            else:
                random_pos = self.sample_random_position()
            
            # Find nearest node
            nearest_node, _ = self.find_nearest_node(nodes, random_pos)
            
            # Steer toward the random position
            new_pos = self.steer(nearest_node.position, random_pos)
            
            # Check if new position is collision-free
            if self.is_collision_free(new_pos) and self.is_path_collision_free(nearest_node.position, new_pos):
                # Create new node
                new_node = Node(new_pos)
                new_node.parent = nearest_node
                nearest_node.children.append(new_node)
                nodes.append(new_node)
                
                # Check if we can connect directly to the goal
                if np.linalg.norm(np.array(new_pos) - np.array(goal_pos)) < self.step_size:
                    if self.is_path_collision_free(new_pos, goal_pos):
                        goal_node.parent = new_node
                        new_node.children.append(goal_node)
                        return self._extract_path(goal_node)
        
        # No path found within max iterations
        return None
    
    def plan_path_rrt_star(self, start_pos, goal_pos):
        """
        Plan a path using the RRT* algorithm.
        
        Args:
            start_pos: Start position as (x, y, z) coordinates
            goal_pos: Goal position as (x, y, z) coordinates
            
        Returns:
            List of positions defining the path, or None if no path found
        """
        # Initialize the tree with the start node
        start_node = Node(start_pos)
        goal_node = Node(goal_pos)
        nodes = [start_node]
        goal_found = False
        best_goal_node = None
        best_goal_cost = float('inf')
        
        # Check if direct path is possible
        if self.is_path_collision_free(start_pos, goal_pos):
            goal_node.parent = start_node
            start_node.children.append(goal_node)
            goal_node.cost = start_node.cost + np.linalg.norm(np.array(goal_pos) - np.array(start_pos))
            return self._extract_path(goal_node)
        
        # Main RRT* loop
        for i in range(self.max_iterations):
            # Sample random position (with bias toward goal)
            if random.random() < self.goal_sample_rate:
                random_pos = goal_pos
            else:
                random_pos = self.sample_random_position()
            
            # Find nearest node
            nearest_node, _ = self.find_nearest_node(nodes, random_pos)
            
            # Steer toward the random position
            new_pos = self.steer(nearest_node.position, random_pos)
            
            # Check if new position is collision-free
            if not (self.is_collision_free(new_pos) and 
                   self.is_path_collision_free(nearest_node.position, new_pos)):
                continue
            
            # Create new node
            new_node = Node(new_pos)
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + np.linalg.norm(np.array(new_pos) - np.array(nearest_node.position))
            
            # Find nearby nodes for rewiring
            nearby_indices = [j for j, node in enumerate(nodes) if 
                             np.linalg.norm(np.array(node.position) - np.array(new_pos)) < self.search_radius]
            
            # Connect to the best parent
            for idx in nearby_indices:
                node = nodes[idx]
                if node == nearest_node:
                    continue
                
                # Check if path is collision-free
                if not self.is_path_collision_free(node.position, new_pos):
                    continue
                
                # Calculate potential cost through this node
                potential_cost = node.cost + np.linalg.norm(np.array(new_pos) - np.array(node.position))
                
                # If better cost, update parent
                if potential_cost < new_node.cost:
                    # Remove from old parent's children
                    if new_node.parent:
                        new_node.parent.children.remove(new_node)
                    
                    # Set new parent
                    new_node.parent = node
                    node.children.append(new_node)
                    new_node.cost = potential_cost
            
            # Add new node to the tree
            if new_node.parent:  # Ensure the node has a valid parent
                new_node.parent.children.append(new_node)
                nodes.append(new_node)
            
                # Rewire the tree
                for idx in nearby_indices:
                    node = nodes[idx]
                    if node == new_node or node == new_node.parent:
                        continue
                    
                    # Check if path is collision-free
                    if not self.is_path_collision_free(new_pos, node.position):
                        continue
                    
                    # Calculate potential cost through the new node
                    potential_cost = new_node.cost + np.linalg.norm(np.array(node.position) - np.array(new_pos))
                    
                    # If better cost, rewire
                    if potential_cost < node.cost:
                        # Remove from old parent's children
                        if node.parent:
                            node.parent.children.remove(node)
                        
                        # Set new parent
                        node.parent = new_node
                        new_node.children.append(node)
                        
                        # Update cost for the node and all its descendants
                        self._update_costs(node, potential_cost)
                
                # Check if we can connect to the goal
                dist_to_goal = np.linalg.norm(np.array(new_pos) - np.array(goal_pos))
                if dist_to_goal < self.step_size and self.is_path_collision_free(new_pos, goal_pos):
                    goal_cost = new_node.cost + dist_to_goal
                    
                    # If it's the best path to goal so far, update
                    if goal_cost < best_goal_cost:
                        # Remove goal from old parent's children if it exists
                        if best_goal_node and best_goal_node.parent:
                            best_goal_node.parent.children.remove(best_goal_node)
                        
                        # Connect goal to new parent
                        goal_node = Node(goal_pos)
                        goal_node.parent = new_node
                        new_node.children.append(goal_node)
                        goal_node.cost = goal_cost
                        
                        best_goal_node = goal_node
                        best_goal_cost = goal_cost
                        goal_found = True
        
        # Return the best path found
        if goal_found:
            return self._extract_path(best_goal_node)
        
        return None
    
    def _update_costs(self, node, new_cost):
        """
        Update the cost of a node and all its descendants.
        
        Args:
            node: Node to update
            new_cost: New cost for the node
        """
        # Update node's cost
        cost_diff = new_cost - node.cost
        node.cost = new_cost
        
        # Recursively update children
        for child in node.children:
            self._update_costs(child, child.cost + cost_diff)
    
    def build_roadmap(self):
        """
        Build a probabilistic roadmap (PRM) of the workspace.
        
        Returns:
            Roadmap as a dictionary of nodes and their connections
        """
        # Sample random valid configurations
        nodes = []
        for _ in range(self.prm_num_samples):
            # Sample position until we find one that's collision-free
            for _ in range(10):  # Limit retry attempts
                pos = self.sample_random_position()
                if self.is_collision_free(pos):
                    nodes.append(Node(pos))
                    break
        
        # Build the graph
        roadmap = defaultdict(list)
        
        # For each node, find nearby nodes and connect if possible
        for i, node in enumerate(nodes):
            # Find nearby nodes
            distances = [(j, np.linalg.norm(np.array(node.position) - np.array(other.position)))
                       for j, other in enumerate(nodes) if j != i]
            
            # Sort by distance
            distances.sort(key=lambda x: x[1])
            
            # Connect to nearest neighbors
            connections = 0
            for j, dist in distances:
                if connections >= self.prm_max_neighbors:
                    break
                
                # Check if path is collision-free
                if self.is_path_collision_free(node.position, nodes[j].position):
                    roadmap[i].append((j, dist))
                    roadmap[j].append((i, dist))
                    connections += 1
        
        # Store the roadmap and nodes
        self.roadmap = roadmap
        self.roadmap_nodes = nodes
        
        return roadmap
    
    def plan_path_prm(self, start_pos, goal_pos):
        """
        Plan a path using the Probabilistic Roadmap (PRM) algorithm.
        
        Args:
            start_pos: Start position as (x, y, z) coordinates
            goal_pos: Goal position as (x, y, z) coordinates
            
        Returns:
            List of positions defining the path, or None if no path found
        """
        # Check if direct path is possible
        if self.is_path_collision_free(start_pos, goal_pos):
            return [start_pos, goal_pos]
        
        # Build or get existing roadmap
        if self.roadmap is None:
            self.build_roadmap()
        
        # Create start and goal nodes
        start_node = Node(start_pos)
        goal_node = Node(goal_pos)
        
        # Add start and goal nodes to the roadmap
        self.roadmap_nodes.append(start_node)
        start_idx = len(self.roadmap_nodes) - 1
        
        self.roadmap_nodes.append(goal_node)
        goal_idx = len(self.roadmap_nodes) - 1
        
        # Connect start and goal to the roadmap
        for i, node in enumerate(self.roadmap_nodes[:-2]):  # Exclude the newly added start and goal
            # Connect start if possible
            dist_to_start = np.linalg.norm(np.array(node.position) - np.array(start_pos))
            if dist_to_start < self.search_radius and self.is_path_collision_free(node.position, start_pos):
                self.roadmap[start_idx].append((i, dist_to_start))
                self.roadmap[i].append((start_idx, dist_to_start))
            
            # Connect goal if possible
            dist_to_goal = np.linalg.norm(np.array(node.position) - np.array(goal_pos))
            if dist_to_goal < self.search_radius and self.is_path_collision_free(node.position, goal_pos):
                self.roadmap[goal_idx].append((i, dist_to_goal))
                self.roadmap[i].append((goal_idx, dist_to_goal))
        
        # Check if start and goal are connected to the roadmap
        if not self.roadmap[start_idx] or not self.roadmap[goal_idx]:
            # Try to connect directly (last resort)
            if self.is_path_collision_free(start_pos, goal_pos):
                return [start_pos, goal_pos]
            return None
        
        # Run A* search to find the shortest path
        path_indices = self._astar_search(start_idx, goal_idx)
        
        if path_indices is None:
            return None
        
        # Convert indices to positions
        path = [self.roadmap_nodes[idx].position for idx in path_indices]
        
        return path
    
    def _astar_search(self, start_idx, goal_idx):
        """
        Perform A* search on the roadmap.
        
        Args:
            start_idx: Index of start node in roadmap_nodes
            goal_idx: Index of goal node in roadmap_nodes
            
        Returns:
            List of node indices defining the path, or None if no path found
        """
        # Priority queue for frontier nodes (cost, node_idx, path_so_far)
        open_set = [(0, start_idx, [start_idx])]
        closed_set = set()
        
        while open_set:
            # Get the node with lowest cost
            curr_cost, curr_idx, curr_path = open_set.pop(0)
            
            # Check if we've reached the goal
            if curr_idx == goal_idx:
                return curr_path
            
            # Skip if already processed
            if curr_idx in closed_set:
                continue
            
            # Add to closed set
            closed_set.add(curr_idx)
            
            # Process neighbors
            for neighbor_idx, edge_cost in self.roadmap[curr_idx]:
                if neighbor_idx in closed_set:
                    continue
                
                # Compute cost to reach neighbor
                g_cost = curr_cost + edge_cost
                
                # Heuristic cost to goal (Euclidean distance)
                h_cost = np.linalg.norm(
                    np.array(self.roadmap_nodes[neighbor_idx].position) - 
                    np.array(self.roadmap_nodes[goal_idx].position)
                )
                
                # Total cost
                f_cost = g_cost + h_cost
                
                # Add to open set
                new_path = curr_path + [neighbor_idx]
                
                # Insert in order of f_cost (lowest first)
                insert_idx = 0
                while insert_idx < len(open_set) and open_set[insert_idx][0] < f_cost:
                    insert_idx += 1
                
                open_set.insert(insert_idx, (f_cost, neighbor_idx, new_path))
        
        # No path found
        return None
    
    def plan_path(self, start_pos, goal_pos):
        """
        Plan a path using the selected method.
        
        Args:
            start_pos: Start position as (x, y, z) coordinates
            goal_pos: Goal position as (x, y, z) coordinates
            
        Returns:
            List of positions defining the path, or None if no path found
        """
        # Check if path planning is enabled
        if not config.PATH_PLANNING_ENABLED:
            # If direct path is collision-free, return it
            if self.is_path_collision_free(start_pos, goal_pos):
                return [start_pos, goal_pos]
            # Otherwise, no path available
            return None
        
        # Try the selected method
        for _ in range(config.MAX_PLANNING_ATTEMPTS):
            try:
                if self.method == PlanningMethod.RRT:
                    path = self.plan_path_rrt(start_pos, goal_pos)
                elif self.method == PlanningMethod.RRT_STAR:
                    path = self.plan_path_rrt_star(start_pos, goal_pos)
                elif self.method == PlanningMethod.PRM:
                    path = self.plan_path_prm(start_pos, goal_pos)
                else:
                    # Unknown method, fall back to RRT
                    path = self.plan_path_rrt(start_pos, goal_pos)
                
                # If path found, smooth it and return
                if path:
                    if config.SMOOTHING_ENABLED:
                        path = self.smooth_path(path)
                    return path
            
            except Exception as e:
                print(f"Path planning error: {str(e)}")
        
        # If all attempts failed, try direct path as fallback
        if self.is_path_collision_free(start_pos, goal_pos):
            return [start_pos, goal_pos]
        
        # No path found
        return None
    
    def _extract_path(self, goal_node):
        """
        Extract the path from start to goal by traversing the tree.
        
        Args:
            goal_node: Goal node
            
        Returns:
            List of positions defining the path
        """
        path = []
        current = goal_node
        
        # Traverse from goal to start
        while current:
            path.append(current.position)
            current = current.parent
        
        # Reverse to get path from start to goal
        path.reverse()
        
        return path
    
    def smooth_path(self, path, max_iterations=100):
        """
        Smooth a path by removing unnecessary waypoints.
        
        Args:
            path: List of positions defining the path
            max_iterations: Maximum number of smoothing iterations
            
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        # Make a copy of the path
        smoothed_path = path.copy()
        
        # Iteratively try to remove points
        for _ in range(max_iterations):
            # Pick two random indices
            i = random.randint(0, len(smoothed_path) - 1)
            j = random.randint(0, len(smoothed_path) - 1)
            
            # Ensure i < j
            if i > j:
                i, j = j, i
            
            # Skip if indices are too close
            if j < i + 2:
                continue
            
            # Check if we can safely skip the points between i and j
            if self.is_path_collision_free(smoothed_path[i], smoothed_path[j]):
                # Remove the points between i and j
                smoothed_path = smoothed_path[:i+1] + smoothed_path[j:]
        
        return smoothed_path
    
    def generate_trajectory(self, path, num_points=None):
        """
        Generate a trajectory by interpolating between path points.
        
        Args:
            path: List of positions defining the path
            num_points: Number of points to generate, defaults to config
            
        Returns:
            List of positions defining the trajectory
        """
        if not path or len(path) < 2:
            return path
        
        if num_points is None:
            num_points = config.TRAJECTORY_POINTS
        
        # Calculate total path length
        total_length = 0
        segments = []
        
        for i in range(len(path) - 1):
            segment_length = np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
            total_length += segment_length
            segments.append((path[i], path[i+1], segment_length))
        
        # Generate evenly spaced points along the path
        trajectory = [path[0]]  # Start with the first point
        remaining_length = total_length
        
        for i in range(1, num_points - 1):  # Generate intermediate points
            target_distance = total_length * i / (num_points - 1)
            
            # Find the segment containing this point
            accumulated_distance = 0
            for start, end, length in segments:
                if accumulated_distance + length >= target_distance:
                    # Calculate interpolation factor
                    t = (target_distance - accumulated_distance) / length
                    
                    # Interpolate position
                    pos = tuple(np.array(start) * (1 - t) + np.array(end) * t)
                    trajectory.append(pos)
                    break
                
                accumulated_distance += length
        
        trajectory.append(path[-1])  # End with the last point
        
        return trajectory


# Function to create obstacles from a depth image
def create_obstacles_from_depth(depth_image, depth_scale, min_depth=None, max_depth=None, 
                               grid_size=20, safety_margin=None):
    """
    Create obstacle representations from a depth image.
    
    Args:
        depth_image: Depth image (height x width)
        depth_scale: Scale factor to convert depth values to mm
        min_depth: Minimum depth to consider (in mm)
        max_depth: Maximum depth to consider (in mm)
        grid_size: Size of grid cells for obstacle creation (in pixels)
        safety_margin: Safety margin to add around obstacles (in mm)
        
    Returns:
        List of obstacle objects
    """
    if depth_image is None:
        return []
    
    # Use defaults from config if not provided
    if min_depth is None:
        min_depth = config.DEPTH_MIN_DISTANCE
    
    if max_depth is None:
        max_depth = config.DEPTH_MAX_DISTANCE
    
    if safety_margin is None:
        safety_margin = config.OBSTACLE_MARGIN
    
    # Initialize obstacle list
    obstacles = []
    
    # Get image dimensions
    height, width = depth_image.shape
    
    # Process the depth image in grid cells
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            # Define cell boundaries
            x_end = min(x + grid_size, width)
            y_end = min(y + grid_size, height)
            
            # Extract depth values in the cell
            cell = depth_image[y:y_end, x:x_end]
            
            # Skip cells with no valid depth
            if np.all(cell == 0):
                continue
            
            # Filter out invalid depths
            valid_depths = cell[(cell > min_depth) & (cell < max_depth)]
            
            if len(valid_depths) == 0:
                continue
            
            # Calculate median depth
            median_depth = np.median(valid_depths)
            
            # Convert to meters
            depth_m = median_depth * depth_scale
            
            # Calculate cell center in image coordinates
            center_x = (x + x_end) // 2
            center_y = (y + y_end) // 2
            
            # Create a box obstacle
            obstacle = BoxObstacle(
                position=(center_x, center_y, depth_m),
                dimensions=(grid_size, grid_size, 50),  # 50mm depth for the box
                safety_margin=safety_margin
            )
            
            obstacles.append(obstacle)
    
    return obstacles


# Test function
def test_path_planning():
    """Test the path planning functionality."""
    # Create a path planner
    planner = PathPlanner(method=PlanningMethod.RRT)
    
    # Add some obstacles
    planner.add_obstacle(SphereObstacle((250, 200, 150), 50))
    planner.add_obstacle(BoxObstacle((150, 150, 100), (50, 50, 50)))
    
    # Define start and goal positions
    start_pos = (50, 50, 50)
    goal_pos = (450, 350, 250)
    
    # Plan a path
    path = planner.plan_path(start_pos, goal_pos)
    
    if path:
        print(f"Path found with {len(path)} waypoints:")
        for i, pos in enumerate(path):
            print(f"  {i}: {pos}")
        
        # Smooth the path
        smoothed_path = planner.smooth_path(path)
        print(f"\nSmoothed path with {len(smoothed_path)} waypoints:")
        for i, pos in enumerate(smoothed_path):
            print(f"  {i}: {pos}")
        
        # Generate trajectory
        trajectory = planner.generate_trajectory(smoothed_path)
        print(f"\nTrajectory with {len(trajectory)} points:")
        for i, pos in enumerate(trajectory):
            print(f"  {i}: {pos}")
    else:
        print("No path found!")


if __name__ == "__main__":
    test_path_planning() 