import copy
import math

map1 = [
    ['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'w', 'w', 'w', 'w', 'w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'w', 'c', 'o', 'c', 'w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'm', 'c', 'c', 'c', 'c', 'w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'w', 'c', 'c', 'c', 'w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'w', 'w', 'w', 'w', 'w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'i', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'w'],
    ['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w']
]

import random

def generate_maze_with_objects(height, width, num_m, num_i,num_s,maze1=map1):
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

# Example usage:
height = 20
width = 20
num_m = 10
num_i = 6
num_s = 1



import numpy as np
import matplotlib.pyplot as plt

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
    plt.show(block=False)

# maze10=generate_maze_with_objects(height, width, num_m, num_i,num_s)
# visualisemaze(maze10)
#
#
# maze3=maze10
#
#
#
#
#
#
#
# ##############################################################################################################
# import pybullet as p
# import time
# import pybullet_data
#
# # open the GUI
# p.connect(p.GUI)
#
# # load files and place them at the offsets
#
#
# # enable real time simulation
# p.setRealTimeSimulation(1)
#
# # define gravity
# p.setGravity(0, 0, -10)
#
# startPos = [0,0,1]
# startOrientation = p.getQuaternionFromEuler([0,0,0])
#
# # define ground
# groundHalfLength = height/2
# groundHalfWidth = width/2
# groundHalfHeight = 0.5
#
# groundIdVisual = p.createVisualShape(p.GEOM_BOX,
#                                   halfExtents=[groundHalfLength, groundHalfWidth, groundHalfHeight],
#                                     rgbaColor=[1, 1, 1, 1],)
#
# groundIdCollision = p.createCollisionShape(p.GEOM_BOX,
#                                   halfExtents=[groundHalfLength, groundHalfWidth, groundHalfHeight])
#
# # define walls
# boxHalfLength = 0.5
# boxHalfWidth = 0.5
# boxHalfHeight = 3
# segmentLength = 5
#
# wallsIdVisual = p.createVisualShape(p.GEOM_BOX,
#                                   halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight],
#                                     rgbaColor=[0.5, 0.5, 0.5, 1],)
#
# wallsIdCollision = p.createCollisionShape(p.GEOM_BOX,
#                                   halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])
#
#
# # define movableobjects
# boxIdVisual = p.createVisualShape(p.GEOM_BOX,
#                                   halfExtents=[0.5, 0.5, 0.5],
#                                   rgbaColor=[0.5, 1, 0.5, 1])
#
# boxIdCollision = p.createCollisionShape(p.GEOM_BOX,
#                                   halfExtents=[0.5, 0.5, 0.5])
#
# # define immovableobjects
# immovableIdVisual = p.createVisualShape(p.GEOM_BOX,
#                                   halfExtents=[0.5, 0.5, 3],
#                                   rgbaColor=[1, 0.5, 0.5, 1])
#
# immovableIdCollision = p.createCollisionShape(p.GEOM_BOX,
#                                   halfExtents=[0.5, 0.5, 3])
#
# # define goal
# goalIdVisual = p.createVisualShape(p.GEOM_BOX,
#                                   halfExtents=[0.5, 0.5, 3],
#                                   rgbaColor=[0, 0, 0, 1])
#
# goalIdCollision = p.createCollisionShape(p.GEOM_BOX,
#                                   halfExtents=[0.5, 0.5, 3])
#
# mass = 1
# visualShapeId = -1
#
# p.createMultiBody(baseMass=0,
#                           baseVisualShapeIndex=groundIdVisual,
#                           baseCollisionShapeIndex=groundIdCollision,
#                           basePosition=[height/2,width/2, 1],
#                           )
#
#
# for i in range(0, height):
#     for j in range(0, width):
#         if maze3[i][j] == "w" :
#             p.createMultiBody(baseMass=0,
#                           baseVisualShapeIndex=wallsIdVisual,
#                           baseCollisionShapeIndex=wallsIdCollision,
#                           basePosition=[i, j, 3],
#                           )
#
#         if maze3[i][j] == "m":
#             p.createMultiBody(baseMass=5,
#                               baseVisualShapeIndex=boxIdVisual,
#                               baseCollisionShapeIndex=boxIdCollision,
#                               basePosition=[i, j, 3],
#                               )
#
#         if maze3[i][j] == "i":
#             p.createMultiBody(baseMass=0,
#                               baseVisualShapeIndex=immovableIdVisual,
#                               baseCollisionShapeIndex=immovableIdCollision,
#                               basePosition=[i, j, 3],
#                               )
#
#         if maze3[i][j] == "o":
#             p.createMultiBody(baseMass=0,
#                               baseVisualShapeIndex=goalIdVisual,
#                               baseCollisionShapeIndex=goalIdCollision,
#                               basePosition=[i, j, 3],
#                               )
#
#         if maze3[i][j] == "s":
#             startx=i
#             starty=j
#

# turtle = p.loadURDF("assets/urdf/most_simple_turtle.urdf", [startx,starty, 3])
#
# distance = 100000
# img_w, img_h = 120, 80
#
# # for debug print out the joints of the turtle
# for j in range(p.getNumJoints(turtle)):
#     print(p.getJointInfo(turtle, j))
#
# forward = 0
# turn = 0




# while (1):
#
#     time.sleep(1. / 240.)
#     keys = p.getKeyboardEvents()
#
#     leftWheelVelocity = 0
#     rightWheelVelocity = 0
#     speed = 50
#
#     for k, v in keys.items():
#
#         if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
#             turn = -0.5
#         if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED)):
#             turn = 0
#         if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
#             turn = 0.5
#         if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED)):
#             turn = 0
#
#         if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED)):
#             forward = 1
#         if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED)):
#             forward = 0
#         if (k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_TRIGGERED)):
#             forward = -1
#         if (k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED)):
#             forward = 0
#
#     rightWheelVelocity += (forward + turn) * speed
#     leftWheelVelocity += (forward - turn) * speed
#
#     p.setJointMotorControl2(turtle, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=100000)
#     p.setJointMotorControl2(turtle, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=100000)
#
#     agent_pos, agent_orn = p.getBasePositionAndOrientation(turtle)
#     yaw = p.getEulerFromQuaternion(agent_orn)[-1]
#
#     xA, yA, zA = agent_pos
#     zA = zA + 0.3 # make the camera a little higher than the robot
#
#     # compute focusing point of the camera
#     xB = xA + math.cos(yaw) * distance
#     yB = yA + math.sin(yaw) * distance
#     zB = zA
#
#     view_matrix = p.computeViewMatrix(
#                         cameraEyePosition=[xA, yA, zA],
#                         cameraTargetPosition=[xB, yB, zB],
#                         cameraUpVector=[0, 0, 1.0]
#                     )
#
#     projection_matrix = p.computeProjectionMatrixFOV(
#                             fov=90, aspect=1.5, nearVal=0.02, farVal=3.5)
#
#     imgs = p.getCameraImage(img_w, img_h,
#                             view_matrix,
#                             projection_matrix, shadow=True,
#                             renderer=p.ER_BULLET_HARDWARE_OPENGL)