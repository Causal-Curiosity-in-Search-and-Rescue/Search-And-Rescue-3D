import math
import numpy as np
import copy
import random
import matplotlib.pyplot as plt

def distance_3d(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def calculate_vector(point1, point2):
    """Calculate the vector from point1 to point2."""
    return np.array(point2) - np.array(point1)

def calculate_cos_angle(vector1, vector2):
    """Calculate the cosine of the angle between two vectors."""
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cos_angle

def quaternion_to_forward_vector(quaternion):
    # Assuming quaternion is in (x, y, z, w) format
    x, y, z, w = quaternion
    # Forward vector
    fx = 2 * (x*z + w*y)
    fy = 2 * (y*z - w*x)
    fz = 1 - 2 * (x*x + y*y)
    return np.array([fx, fy, fz])

def generate_maze_with_objects( height, width, num_m, num_i,num_s):
    maze1 = generate_maze(height,width)
    maze = copy.deepcopy(maze1)
 
    # Generate random positions for 'm'
    for _ in range(num_m):
        while True:
            row = random.randint(1, height - 2)
            col = random.randint(1, width - 2)
            if maze[row][col] == 'c':
                maze[row][col] = 'm'
                break
 
    # Generate random positions for 'i'
    for _ in range(num_i):
        while True:
            row = random.randint(1, height - 2)
            col = random.randint(1, width - 2)
            if maze[row][col] == 'c':
                maze[row][col] = 'i'
                break
    # Generate random positions for 's'
    for _ in range(num_s):
        while True:
            row = random.randint(1, height - 2)
            col = random.randint(1, width - 2)
            if maze[row][col] == 'c':
                maze[row][col] = 's'
                break
 
    return maze

def visualisemaze(maze):
    # Convert maze to numpy array for visualization
    maze_np = np.array([[1 if cell == 'c' else 2 if cell == 'm' else 3 if cell == 'i' else 4 if cell == 'o' else 5 if cell == 's' else 0 for cell in row] for row in maze])

    # Define custom color map
    cmap = plt.cm.viridis
    cmap.set_over('orange')

    plt.imshow(maze_np, cmap=cmap, interpolation='nearest')

    # Customizing color bar for legend
    plt.colorbar(ticks=[0, 1, 2, 3, 4, 5], label='Legend', values=[0, 1, 2, 3, 4, 5], format=plt.FuncFormatter(lambda val, loc: ['walls', 'free space', 'moveable objects', 'immovable objects', 'goal', 'starting position'][int(val)]))

    plt.axis('off')
    # plt.show(block=False)
    plt.savefig('2d_test_env.png')
    
def generate_maze(height, width):
    maze = [['w' if i == 0 or i == height - 1 or j == 0 or j == width - 1 else 'c' for j in range(width)] for i in range(height)]
    maze_with_array = place_array_in_middle(maze)
    maze_with_u = place_u_near_walls(maze_with_array)
    return maze_with_u
 
def place_array_in_middle(maze):
    middle_row = len(maze) // 2
    middle_col = len(maze[0]) // 2
 
    array_to_place = [
        ['u','u', 'u', 'u', 'u', 'u', 'u','u'],
        ['u','u', 'r', 'r', 'r', 'r', 'r','u'],
        ['u','u', 'm', 'u', 'o', 'u', 'r','u'],
        ['u','u', 'm', 'u', 'u', 'u', 'r','u'],
        ['u','u', 'r', 'u', 'u', 'u', 'r','u'],
        ['u','u', 'r', 'r', 'r', 'r', 'r','u'],
        ['u','u', 'u', 'u', 'u', 'u', 'u','u']
    ]
 
    for i in range(len(array_to_place)):
        for j in range(len(array_to_place[0])):
            maze[middle_row - 2 + i][middle_col - 3 + j] = array_to_place[i][j]
 
    return maze
 
 
def place_u_near_walls(maze):
    height = len(maze)
    width = len(maze[0])
 
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if maze[i][j] == 'c':
                adjacent_cells = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for x, y in adjacent_cells:
                    if maze[x][y] == 'w':
                        maze[i][j] = 'u'
 
    return maze