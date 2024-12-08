
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from create_plan_msgs.srv import CreatePlan
from nav2_simple_commander.robot_navigator import BasicNavigator
import heapq


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")

        # Initialize BasicNavigator to get the global costmap
        self.basic_navigator = BasicNavigator()

        # Create a service "create_plan"
        self.srv = self.create_service(CreatePlan, 'create_plan', self.create_plan_cb)

    def create_plan_cb(self, request, response):
        # Get start and goal poses
        start_pose = request.start
        goal_pose = request.goal
        time_now = self.get_clock().now().to_msg()

        # Retrieve the global costmap
        global_costmap = self.basic_navigator.getGlobalCostmap()
        costmap = global_costmap.data
        width = global_costmap.info.width
        height = global_costmap.info.height
        resolution = global_costmap.info.resolution
        origin = global_costmap.info.origin

        # Convert the costmap data into a 2D array
        costmap_2d = [
            costmap[i * width:(i + 1) * width]
            for i in range(height)
        ]

        # Plan the path using A* algorithm
        response.path = a_star_planner(
            start_pose, goal_pose, costmap_2d, resolution, origin, time_now
        )
        return response


def a_star_planner(start, goal, costmap, resolution, origin, time_now):
    """
    A* algorithm to compute a path avoiding obstacles.

    Args:
        start (PoseStamped): Start pose.
        goal (PoseStamped): Goal pose.
        costmap (list of lists): 2D costmap grid.
        resolution (float): Resolution of the grid (meters per cell).
        origin (PoseStamped): Origin of the costmap (bottom-left corner).
        time_now (Time): Current ROS2 time.

    Returns:
        Path: Computed path as a `nav_msgs.msg.Path`.
    """
    path = Path()
    path.header.frame_id = goal.header.frame_id
    path.header.stamp = time_now

    # Convert start and goal to grid coordinates
    start_grid = (
        int((start.pose.position.x - origin.position.x) / resolution),
        int((start.pose.position.y - origin.position.y) / resolution),
    )
    goal_grid = (
        int((goal.pose.position.x - origin.position.x) / resolution),
        int((goal.pose.position.y - origin.position.y) / resolution),
    )

    def heuristic(a, b):
        """Calculate the Euclidean distance as the heuristic."""
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    # Priority queue for A*
    open_set = []
    heapq.heappush(open_set, (0, start_grid))
    came_from = {}
    g_score = {start_grid: 0}

    while open_set:
        # Get the current node with the lowest cost
        _, current = heapq.heappop(open_set)

        # If goal is reached
        if current == goal_grid:
            # Reconstruct path
            while current in came_from:
                x, y = current
                pose = PoseStamped()
                pose.pose.position.x = x * resolution + origin.position.x
                pose.pose.position.y = y * resolution + origin.position.y
                pose.header.stamp = time_now
                pose.header.frame_id = goal.header.frame_id
                path.poses.append(pose)
                current = came_from[current]

            path.poses.reverse()
            return path

        # Get neighbors
        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]

        for neighbor in neighbors:
            # Skip if neighbor is out of bounds or in obstacle
            if (neighbor[0] < 0 or neighbor[1] < 0 or
                    neighbor[0] >= len(costmap) or neighbor[1] >= len(costmap[0]) or
                    costmap[neighbor[1]][neighbor[0]] > 50):  # Threshold for obstacles
                continue

            # Calculate tentative g-score
            tentative_g_score = g_score[current] + resolution

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # Update the path
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal_grid)
                heapq.heappush(open_set, (f_score, neighbor))

    # If no path is found, return an empty path
    return path


def main(args=None):
    rclpy.init(args=args)
    path_planner_node = PathPlannerNode()

    try:
        rclpy.spin(path_planner_node)
    except KeyboardInterrupt:
        pass

    path_planner_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()








# Industrial Project Part 4: PCA-Based Spectral Data Analysis

This project processes and analyzes spectral data using Principal Component Analysis (PCA). It supports multiple tasks such as `structure_oil`, `structure_charring`, `oil_charring`, and `cracking`. Tasks are configured via a `config2.yaml` file, enabling flexibility and reproducibility.

## Features

- **PCA Analysis**: Performs dimensionality reduction on spectral data and visualizes the explained variance and principal components.
- **Task-Based Execution**: Supports multiple tasks (e.g., `structure_oil`, `structure_charring`) specified in the configuration file.
- **Configurable Parameters**: Input paths, number of PCA components, and task selection are managed via `config2.yaml`.
- **Custom Visualizations**:
  - Explained variance bar charts.
  - False-colored PCA component images.
  - Loading vectors for selected principal components.

## Installation

### Requirements

- Python 3.8 or higher
- The following Python libraries:
  - `numpy`
  - `matplotlib`
  - `PyYAML`
  - `scikit-learn`
  - `opencv-python`

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sonainjameel/Industrial_Project_2024_Part_04.git
   cd Industrial_Project_2024_Part_04
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Input Data**: Ensure your `.hdr` and `.raw` files are placed in the folders specified in `config2.yaml`.

## Configuration

The project uses a `config2.yaml` file to define the tasks and their parameters. Below is an example configuration:

```yaml
tasks:
  - name: structure_oil
    dir_path: "path/to/oil/data"
    components: 10

  - name: structure_charring
    dir_path: "path/to/charring/data"
    components: 10

  - name: oil_charring
    dir_path: "path/to/oil_charring/data"
    components: 10

  - name: cracking
    dir_path: "path/to/cracking/data"
    components: 10
```

## Usage

1. **Run All Tasks**:
   To process all tasks defined in the `config2.yaml`:
   ```bash
   python3 pca_analysis.py
   ```

2. **Run a Specific Task**:
   To run only a specific task (e.g., `structure_oil`):
   ```bash
   python3 pca_analysis.py --task structure_oil
   ```

## Example Outputs

- **Structure Oil Analysis**:
  - PCA plots showing oil levels across different structures.
  - Loading vector visualizations with annotated spectral regions.

- **Charring Analysis**:
  - Visualizations of charred regions in PCA space.

## Project Structure

```
Industrial_Project_2024_Part_04/
├── config2.yaml               # Configuration file for tasks and parameters
├── pca_analysis.py    # Script for PCA processing and plotting
├── requirements.txt          # Required libraries
```

## Available Tasks

- **structure_oil**: Analyzes spectral data for oil content across structures.
- **structure_charring**: Examines spectral data for charring levels.
- **oil_charring**: Investigates the interaction of oil content and charring.
- **cracking**: Focuses on spectral data related to material cracking.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with a clear message.
4. Submit a pull request for review.

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to Sonain, Kasem, and Turab for their efforts, and to Joni Hyttinen and Prof. Markku Keinänen for their guidance and support throughout the project.
