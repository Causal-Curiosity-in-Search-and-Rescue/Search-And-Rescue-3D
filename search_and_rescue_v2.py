import logging
# Setup basic configuration for logging
logging.basicConfig(filename='debug.log', level=logging.INFO,filemode='w', format='%(asctime)s %(message)s')
import warnings
warnings.filterwarnings("ignore")
import gym
import json
import os 
import numpy as np
import pybullet as p
from gym import spaces
import math
from joblib import load
from collections import deque
import time
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.network import BayesianNetwork
from causalnex.structure.notears import from_pandas_lasso
import pandas as pd 
from stable_baselines3 import PPO,A2C
# Custom imports
from digital_mind import EnvironmentObjectsManager
from preprocessing import normalise_textures,get_texture_features
from helpers import distance_3d,calculate_vector,calculate_cos_angle,quaternion_to_forward_vector,generate_maze_with_objects,visualisemaze
import pickle
import optuna 
from stable_baselines3.common.evaluation import evaluate_policy
from gym.envs.registration import register
import wandb
import imageio
import pdb
# import cv2

# LOAD THE URDF FILES AND TEXTURES
BASE_PATH = os.path.join(os.getcwd(),"resources")

# LOAD COMPUTERVISION MODELS
CV_MODEL = load(f"{BASE_PATH}/models/unsup_txture_clsf_rf.joblib")
CV_SCALER = load(f"{BASE_PATH}/models/unsup_txture_clsf_scaler.joblib")
CV_THRESHOLD = 0.7
MOVABILITY_THRESHOLD = 0.7
SCALER = load(f"{BASE_PATH}/models/scaler.joblib")

# Initialize Digital MIND
ENV_MANAGER = EnvironmentObjectsManager()

AGENT_ACTION_LEN = 30
p.connect(p.GUI)
# p.connect(p.DIRECT)

height = 20
width = 20
num_m = 2 # Movable
num_i = 15 # Immovable
num_s = 1 # Start Positions
n_texture_classes = 2
n_objects = num_m + 2 + num_i
map_plan = generate_maze_with_objects(height, width, num_m, num_i, num_s) 
with open('maze_plan.pkl','wb') as file:
    pickle.dump(map_plan,file)
visualisemaze(map_plan)

# TRAINING the RL 
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

def create_or_update_object(detected_object_id,texture_class,detected_x,detected_y,detected_z,detected_texture):
    attributes = {
        'texture': detected_texture,
        'x': detected_x,
        'y': detected_y,
        'z': detected_z
    }
    ENV_MANAGER.update_or_create_object(object_id=detected_object_id,texture_class = texture_class,**attributes)

class SearchAndRescueEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}
  
    def __init__(self):
        super(SearchAndRescueEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # Forward, Left, Right
        self.observation_space = spaces.Dict({
            'positional_data': spaces.Box(low=np.inf, high=np.inf, shape=(6+AGENT_ACTION_LEN,), dtype=np.float32),
            'object_data': spaces.Box(low=-np.inf, high=np.inf, shape=(n_objects, 5), dtype=np.float32),
            'collision_info': spaces.Discrete(6)  # 0: No collision, 1: Collided with wall, 2: Collided with room,,  3: Collided with immovable, 4: Collided with movable, 5: collision with goal
        })

        # Initial Params
        self.movability_dict = {0:None,1:None}
        self.movability_predictions = {}
        self.causal_interaction_count = 0
        self.causal_probablity_dict = {}
        sorted_obj_ids = [0,1]
        for objectID in sorted_obj_ids:
            self.movability_predictions[objectID] = []
            self.causal_probablity_dict[objectID] = 0 # Set 0 for default and do + for the actual 
    
    def create_walls(self,map_plan):
        # define ground
        groundHalfLength = height/2
        groundHalfWidth = width/2
        groundHalfHeight = 1

        groundIdVisual = p.createVisualShape(p.GEOM_BOX,
                                          halfExtents=[groundHalfLength, groundHalfWidth, groundHalfHeight],
                                            rgbaColor=[1, 1, 1, 1],)

        groundIdCollision = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=[groundHalfLength, groundHalfWidth, groundHalfHeight])
        p.createMultiBody(baseMass=0,
                                  baseVisualShapeIndex=groundIdVisual,
                                  baseCollisionShapeIndex=groundIdCollision,
                                  basePosition=[height/2,width/2, -1],
                                  )

        self.wall_ids = []  # Store wall IDs in an attribute for later access
        self.room_ids = []  # Store Room IDS
        # define walls
        boxHalfLength = 0.5
        boxHalfWidth = 0.5
        boxHalfHeight = 1

        wallsIdVisual = p.createVisualShape(p.GEOM_BOX,
                                            halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight],
                                            rgbaColor=[0.5, 0.5, 0.5, 1], )

        wallsIdCollision = p.createCollisionShape(p.GEOM_BOX,
                                                  halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])
        
        self.room_wall_positions = {'north': [], 'south': [], 'east': [], 'west': []}
        for i in range(0, height):
            for j in range(0, width):
                if map_plan[i][j] == "w":
                    wall_id = p.createMultiBody(baseMass=0,
                                      baseVisualShapeIndex=wallsIdVisual,
                                      baseCollisionShapeIndex=wallsIdCollision,
                                      basePosition=[i, j, 1],
                                      )
                    self.wall_ids.append(wall_id)
                if map_plan[i][j] == "r":
                    room_id = p.createMultiBody(baseMass=0,
                                      baseVisualShapeIndex=wallsIdVisual,
                                      baseCollisionShapeIndex=wallsIdCollision,
                                      basePosition=[i, j, 1],
                                      )
                    self.room_ids.append(room_id)
                    # Determine the direction of the wall and add to room_wall_positions
                    if i == 0 or i < height / 2:  # Assuming 'north' is towards the top of the map
                        self.room_wall_positions['north'].append((i, j))
                    if i == height - 1 or i >= height / 2:  # 'south'
                        self.room_wall_positions['south'].append((i, j))
                    if j == 0 or j < width / 2:  # 'west'
                        self.room_wall_positions['west'].append((i, j))
                    if j == width - 1 or j >= width / 2:  # 'east'
                        self.room_wall_positions['east'].append((i, j))
                    
        # Calculate midpoints for internal rooms
        self.room_wall_midpoints = {}
        for direction, positions in self.room_wall_positions.items():
            if positions:  # Check if there are any room walls in this direction
                avg_x = sum(pos[0] for pos in positions) / len(positions)
                avg_y = sum(pos[1] for pos in positions) / len(positions)
                self.room_wall_midpoints[direction] = (avg_x, avg_y, 1)  # Assuming walls are all at height 1
                    
        # Get Midpoint of the Walls for each direction and the midpoint of the environment
        self.wall_midpoints = {'north': (height / 2, width - 1, 1), 'south': (height / 2, 0, 1), 
                            'east': (height - 1, width / 2, 1), 'west': (0, width / 2, 1)}

        self.environment_midpoint = (height / 2, width / 2, 1)
        
        self.scaling_factor = self.calculate_half_length_scaling() # Will Experiment with global or local
        
    def calculate_half_length_scaling(self):
        """
        Half Length Scaling Factor can be good for Local Exploration
        """
        wall_midpoint = self.wall_midpoints['north'] # can use any wall
        env_midpoint = self.environment_midpoint
        half_length = ((wall_midpoint[0] - env_midpoint[0]) ** 2 + 
                       (wall_midpoint[1] - env_midpoint[1]) ** 2 + 
                       (wall_midpoint[2] - env_midpoint[2]) ** 2) ** 0.5
        return half_length

    def calculate_full_length_scaling(self):
        """
        Full Length Scaling Factor can be good for a Global Perspective of the Env.
        """
        north_midpoint = self.wall_midpoints['north']
        south_midpoint = self.wall_midpoints['south']
        full_length = ((north_midpoint[0] - south_midpoint[0]) ** 2 + 
                       (north_midpoint[1] - south_midpoint[1]) ** 2 + 
                       (north_midpoint[2] - south_midpoint[2]) ** 2) ** 0.5
        return full_length
    
    def create_obstacles(self,map_plan):
        # define movableobjects
        object_ids = []
        boxIdVisual = p.createVisualShape(p.GEOM_BOX,
                                          halfExtents=[0.5, 0.5, 0.5],
                                          #rgbaColor=[0.5, 1, 0.5, 1]
                                          )

        boxIdCollision = p.createCollisionShape(p.GEOM_BOX,
                                                halfExtents=[0.5, 0.5, 0.5])

        # define immovableobjects
        immovableIdVisual = p.createVisualShape(p.GEOM_BOX,
                                                halfExtents=[0.5, 0.5, 0.5],
                                                #rgbaColor=[1, 0.5, 0.5, 1]
                                                )

        immovableIdCollision = p.createCollisionShape(p.GEOM_BOX,
                                                      halfExtents=[0.5, 0.5, 0.5])

        goalIdVisual = p.createVisualShape(p.GEOM_BOX,
                                          halfExtents=[0.5, 0.5, 3],
                                          rgbaColor=[0, 1, 0, 1])
        goalIDCollision = p.createCollisionShape(p.GEOM_BOX,
                                        halfExtents=[0.5, 0.5, 0.5])
        
        
        for i in range(0, height):
            for j in range(0, width):
                if map_plan[i][j] == "m":
                    #id_num = int(str(i) + str(j))
                    id_num = p.createMultiBody(baseMass=1,
                                      baseVisualShapeIndex=boxIdVisual,
                                      baseCollisionShapeIndex=boxIdCollision,
                                      basePosition=[i, j, 0.5],
                                      )
                    p.changeVisualShape(id_num, -1, textureUniqueId=self.CRACKED_1)
                    object_ids.append(id_num)
                    self.movable_obj_ids.append(id_num)

                if map_plan[i][j] == "i":
                    #id_num = int(str(i) + str(j))
                    id_num = p.createMultiBody(baseMass=0,
                                      baseVisualShapeIndex=immovableIdVisual,
                                      baseCollisionShapeIndex=immovableIdCollision,
                                      basePosition=[i, j, 0.5],
                                      )
                    p.changeVisualShape(id_num, -1, textureUniqueId=self.MARBLED_2)
                    object_ids.append(id_num)
                    self.immovable_obj_ids.append(id_num)

                if map_plan[i][j] == "o":
                    
                    self.goal_id = p.createMultiBody(baseMass=0,
                                      baseVisualShapeIndex=goalIdVisual,
                                      baseCollisionShapeIndex=goalIDCollision,
                                      basePosition=[i, j, 3],
                                      )

        return object_ids # returns the unique id for each obstacle created
    
    def initialized_objects_position(self,objectIDs):
        # Dictionary to store positions for each object ID
        objectPositions = {}
        if not isinstance(objectIDs,int):
            for objID in objectIDs:
                pos, orn = p.getBasePositionAndOrientation(objID)
                # pos = (int(pos[0]),int(pos[1]),int(pos[2]))
                objectPositions[objID] = pos
        else:
            pos, orn = p.getBasePositionAndOrientation(objectIDs)
            objectPositions = pos
        return objectPositions
    
    def translate_action(self,action):
        move = 0
        turn = 0
        speed = 7
        leftWheelVelocity = 0
        rightWheelVelocity = 0
        
        if action == 0: # forward 
            move = 1
        # if action == 1: #backward 
        #     move = -1
        if action == 1: # turn right
            turn = -0.5
        if action == 2: #turn left 
            turn = 0.5
            
        rightWheelVelocity += (move + turn) * speed
        leftWheelVelocity += (move - turn) * speed
        p.setJointMotorControl2(self.TURTLE, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1000)
        p.setJointMotorControl2(self.TURTLE, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1000)
        p.stepSimulation()
        
        if action == 0: # forward 
            move = 0
        # if action == 1: #backward 
        #     move = 0
        if action == 1: # turn right
            turn = 0
        if action == 2: #turn left 
            turn = 0
        
        rightWheelVelocity += (move + turn) * speed
        leftWheelVelocity += (move - turn) * speed
        p.setJointMotorControl2(self.TURTLE, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1000)
        p.setJointMotorControl2(self.TURTLE, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1000)
        p.stepSimulation()
            
    def setup_agent(self):
        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.TURTLE)
        yaw = p.getEulerFromQuaternion(agent_orn)[-1]
        xA, yA, zA = agent_pos
        zA = zA + 0.3 # make the camera a little higher than the robot

        # compute focusing point of the camera
        xB = xA + math.cos(yaw) * self.distance
        yB = yA + math.sin(yaw) * self.distance
        zB = zA
        
        return (xA,yA,zA), (xB,yB,zB), agent_orn # Agent Position, Camera Position and Orientation of Robot
    
    def start_sensing_module_and_initializing_digital_mind(self): # to be used in step and reset
        robot_position, camera_position, robot_orientation = self.setup_agent()
        img_w, img_h = 120, 80
        view_matrix = p.computeViewMatrix(
                        cameraEyePosition=[robot_position[0], robot_position[1], robot_position[2]],
                        cameraTargetPosition=[camera_position[0], camera_position[1], camera_position[2]],
                        cameraUpVector=[0, 0, 1.0]
                    )

        projection_matrix = p.computeProjectionMatrixFOV(
                                fov=90, aspect=1.5, nearVal=0.02, farVal=3.5)

        width, height, rgbImg, depthImg, segImg = p.getCameraImage(img_w, img_h,
                                view_matrix,
                                projection_matrix, shadow=True,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        self.rgbImg = rgbImg
        self.rgbImgWidth = width
        self.rgbImgHeight = height
        
        ###Get Segmented IDs
        img = np.reshape(rgbImg, (height, width, 4)) * 1. / 255.
        gray_scale_image = np.mean(img, axis=2)  # Convert RGB to grayscale by averaging the color channels

        #Get features
        features = get_texture_features(gray_scale_image)
        features = normalise_textures(features,SCALER)

        #'segImg' is the segmentation image obtained from 'p.getCameraImage'
        unique_ids = np.unique(segImg)

        # Filter out the ID for the ground plane or any other IDs not relevant
        unique_ids = [uid for uid in unique_ids if uid not in [self.PLANE, -1]]
        
        sensing_info = []
        
         # Assuming you want to know the positions of these unique IDs
        for uid in unique_ids:
            if uid in self.objectIDs:
                pos, orn = p.getBasePositionAndOrientation(uid)
                pos = (int(pos[0]), int(pos[1]), int(pos[2])) # position of the object
                predict_array = np.array([[features['contrast'], features['dissimilarity'], features['homogeneity'],
                                        features['energy'], features['correlation'], features['ASM']]])

                #Get euclidean distance to object
                distance = distance_3d(pos, self.robot_position)
                current_visual_prediction = None
                # Calculate the vector to the object
                vector_to_object = calculate_vector(self.robot_position, pos)
                # Calculate the cosine of the angle between your orientation and the vector to the object
                cos_angle = calculate_cos_angle(quaternion_to_forward_vector(robot_orientation), vector_to_object)
                
                object_information = {
                        'contrast': features['contrast'],
                        'dissimilarity': features['dissimilarity'],
                        'homogeneity': features['homogeneity'],
                        'energy': features['energy'],
                        'correlation': features['correlation'],
                        'ASM': features['ASM'],
                        'distance':distance,
                        'cos_angle':cos_angle
                    }
                object_df = pd.DataFrame([object_information])
                scaled_data = CV_SCALER.transform(object_df)
                predictions = CV_MODEL.predict(scaled_data)
                if predictions[0] in [2,1]:
                    current_visual_prediction = 1
                else:
                    current_visual_prediction = 0
                sorted_obj_id = self.sort_obj_ids_truth(uid)
                thresholded_prediction = self.apply_thresholding(sorted_obj_id,current_visual_prediction)
                # thresholded_prediction = current_visual_prediction
                classified_image = "Cracked" if (thresholded_prediction == 1) else "Grooved"
                print(f'[INFO] Classified Texture Class : {classified_image} - UID : {uid} - Position : {pos} -  {thresholded_prediction} - {distance} - {cos_angle}')
                
                #On getting close to object for inspection, get the predicted class
                if distance<=1.8 and cos_angle < -0.50: # Robot is facing the object 
                    # Updated Vision Texture Classification 
                    thresholded_prediction = self.apply_thresholding(sorted_obj_id,current_visual_prediction)
                    sensing_info.append(
                        {
                            'uid':sorted_obj_id,
                            'obj_id':uid,
                            'distance': distance,
                            'cos_angle':cos_angle,
                            'is_facing_object':True,
                            'object_position_has_changed':self.has_object_moved_due_to_robot(uid),
                            'current_visual_prediction':thresholded_prediction
                        }
                    )
                    
                else:
                    thresholded_prediction = None
                    sensing_info.append(
                        {
                            'uid':sorted_obj_id,
                            'obj_id':uid,
                            'distance': distance,
                            'cos_angle':cos_angle,
                            'is_facing_object':False,
                            'object_position_has_changed':None,
                            'current_visual_prediction':None
                        }
                    )
                
                create_or_update_object(sorted_obj_id,thresholded_prediction, pos[0], pos[1], pos[2],features)
        return sensing_info
    
    def sort_obj_ids_truth(self,uid):
        if uid in self.movable_obj_ids:
            return 1
        # elif uid in self.immovable_obj_ids:
        else:
            return 0
    
    def apply_thresholding(self,uid,current_prediction):
        self.visual_predictions[uid].append(current_prediction)
        # print(f'[DEBUG] Length of Current Visual Predictions for {uid} then {len(self.visual_predictions[uid])}')
        prediction_prob = np.mean(self.visual_predictions[uid])
        # print(f"[DEBUG] Mean of visual Predictions for {uid} then {prediction_prob}")
        if len(self.visual_predictions[uid]) > 1:
            logging.info(f"[DEBUG] Mean of visual Predictions for {uid} then {prediction_prob}")
            if prediction_prob > CV_THRESHOLD:
                return 1
            else:
                return 0
        else:
            return None # This allows for Predictions to be more stable
    
    def has_object_moved_due_to_robot(self, uid, movement_threshold=0.05):
        # Get the new position of the object and the robot
        new_object_position, _ = p.getBasePositionAndOrientation(uid)
        new_robot_position = self.robot_position  # Assuming this is updated elsewhere before this method is called

        # Retrieve the initial positions from your records
        initial_object_position = self.objectPositions[uid]
        initial_robot_position = self.previous_position

        # Calculate the displacement distances
        object_displacement = np.linalg.norm(np.array(initial_object_position) - np.array(new_object_position))
        robot_displacement = np.linalg.norm(np.array(initial_robot_position) - np.array(new_robot_position))
        # Determine if the object has moved significantly (more than the threshold)
        object_moved_significantly = object_displacement > movement_threshold

        # Now, check if the robot's displacement is aligned with the object's displacement
        # You could use vector comparison or simply check if both moved noticeably
        aligned_movement = False
        if object_moved_significantly and robot_displacement > movement_threshold:
            # Calculate vectors to see if they are in the same direction; this is a simplified example
            object_movement_vector = np.array(new_object_position) - np.array(initial_object_position)
            robot_movement_vector = np.array(new_robot_position) - np.array(initial_robot_position)
            
            # Normalize vectors and calculate dot product to check alignment
            norm_obj_vector = object_movement_vector / np.linalg.norm(object_movement_vector)
            norm_robot_vector = robot_movement_vector / np.linalg.norm(robot_movement_vector)
            dot_product = np.dot(norm_obj_vector, norm_robot_vector)

            # Check if the movement direction is similar (dot product close to 1)
            if dot_product > 0.5:  # Adjust this value based on how strict you want this check to be
                aligned_movement = True

        # The object is considered moved due to the robot if it moved significantly and in alignment with the robot's movement
        return object_moved_significantly and aligned_movement
    
    def control_movability_update(self,uid):
        if len(self.movability_predictions[uid]) > 1: # number of minimum interactions for it to update of movability
            movability_prob =  np.mean(self.movability_predictions[uid])
            if movability_prob > MOVABILITY_THRESHOLD:
                self.movability_dict[uid] = 1
                return 1
            else:
                self.movability_dict[uid] = 0
                return 0
        else:
            return None
    
    def update_moability_in_digital_mind_using_last_action(self,sensing_info):
        for idx in range(len(sensing_info)):
            info = sensing_info[idx]
            # if self.last_action == 0: # last action was forward
            if info['is_facing_object']:
                if info['object_position_has_changed'] == False:
                    for obj_id, obj in ENV_MANAGER.objects.items():
                        if obj.id == info['uid']:
                            if obj.movability == None:
                                self.movability_predictions[info['uid']].append(0)
                                # obj.movability = self.control_movability_update(info['uid'])
                            
                if info['object_position_has_changed']:
                    for obj_id, obj in ENV_MANAGER.objects.items():
                        if obj.id == info['uid']:
                            if obj.movability == None:
                                self.movability_predictions[info['uid']].append(1)
                                # obj.movability = self.control_movability_update(info['uid'])
        
        self.check_movability_predictions_and_update_digital_mind()
    
    def check_movability_predictions_and_update_digital_mind(self):
        for obj_id, obj in ENV_MANAGER.objects.items():
            if obj.id == 0:
                obj.movability = self.control_movability_update(0)
            
            if obj.id == 1:
                obj.movability = self.control_movability_update(1)

    def store_causal_probability(self,df):
        texture_0_movability = df[df['Texture'] == 0]['Movability_1'].iloc[0]
        texture_1_movability = df[df['Texture'] == 1]['Movability_1'].iloc[0]
        for obj_id, obj in ENV_MANAGER.objects.items():
            if obj.texture_class == 0:
                self.causal_probablity_dict[0] = int(texture_0_movability)
                obj.casual_probability = int(texture_0_movability)
            else:
                self.causal_probablity_dict[1] = int(texture_1_movability)
                obj.casual_probability = int(texture_1_movability)
             
    def casual_reasoning_for_object_movability(self): # to be called and used in step
        texture = np.array([0,1])
        movability = np.array([1, 1])
        state = 0
        for obj_id, obj in ENV_MANAGER.objects.items():
            if obj.movability == None:
                movability = np.array([1, 1])# it will set everything to be movable on the truth table
            else:
                if self.movability_dict[0] == None:
                    if obj.texture_class == 0:
                        if obj.movability != None:
                            # print(f'For obj.id : {obj.id} - Texture : {obj.texture_class} - movability : {obj.movability}')
                            self.causal_interaction_count += 1
                            self.movability_dict[0] = int(obj.movability)
                        else:
                            self.movability_dict[0] = None

                if self.movability_dict[1] == None:
                    if obj.texture_class == 1:
                        if obj.movability != None:
                            self.causal_interaction_count += 1
                            # print(f'For obj.id : {obj.id} - Texture : {obj.texture_class} - movability : {obj.movability}')
                            self.movability_dict[1] = int(obj.movability)
                        else:
                            self.movability_dict[1] = None
                   
                logging.info(f'Movability Dictionary: {self.movability_dict}')                
                
                if (self.movability_dict[0] != None) and (self.movability_dict[1] != None):
                    logging.info(f'[METRIC] Causal Graph Created - Number of Interactions Required : {self.causal_interaction_count}') # Evaluation Metric
                    logging.info(f"[METRIC] Minimum Number of Interactions for Movable Objects : {len(self.movability_predictions[1])}")
                    logging.info(f"[METRIC] Minimum Number of Interactions for Immovable Objects : {len(self.movability_predictions[0])}")
                    movability = np.array([self.movability_dict[0],self.movability_dict[1]])
                    state = 1
                else:
                    movability = np.array([1, 1])# it will set everything to be movable on the truth table
        try:
            n = 100
            aug_texture = np.tile(texture,n)
            aug_movability = np.tile(movability,n)
            
            df = pd.DataFrame({
                'Texture':aug_texture,
                'Movability':aug_movability
            })
            causal_model = from_pandas_lasso(df,beta=0.001,w_threshold=0.1)
            viz = plot_structure(causal_model,graph_attributes={"scale": "1.0"},all_node_attributes=NODE_STYLE.WEAK,all_edge_attributes=EDGE_STYLE.WEAK,prog='fdp')
            viz.draw('causal_graph.png', format='png')
            
            bn = BayesianNetwork(causal_model)
            bn = bn.fit_node_states(df)
            bn = bn.fit_cpds(df, method="BayesianEstimator", bayes_prior="K2")
            # Predicting the probabilities
            predictions = bn.predict_probability(df, 'Movability')
            # logging.info(f"Causal Probability Predicted : {predictions['Movability_1']}")
            
            binary_predictions = predictions['Movability_1'] > 0.5  # Adjust based on actual column names in predictions
            
            combined_df =  pd.concat([df, binary_predictions], axis=1)
            self.store_causal_probability(combined_df)
            if state == 1:
                logging.info("[SCCS] Causal relation was Created using Interactions ")
        except Exception as Error:
            logging.info("[ERR] Causal Relation Could not be Created Due to False Positives in Predictions Leading to Invalid Interactions")
        
    def get_texture_and_movability(self,objectID):
        sorted_obj_id = self.sort_obj_ids_truth(objectID)
        for obj_id, obj in ENV_MANAGER.objects.items():
            if obj.id == sorted_obj_id:
                if obj.texture_class != None:
                    return  int(obj.texture_class), int(obj.casual_probability),obj.initial_position
                return -1,int(obj.casual_probability),obj.initial_position
        return -1,-1,(None,None,None)

    def check_collision_with_walls(self):
        collision_info = {
            'has_collided': False,
            'collided_wall': None
        }

        # Retrieve all contact points
        contact_points = p.getContactPoints(bodyA=self.TURTLE)

        for contact in contact_points:
            # Check if the contact involves the robot and any of the walls
            if contact[2] in self.wall_ids:  # contact[2] is the unique ID of the second object in the collision
                collision_info['has_collided'] = True
                collision_info['collided_wall'] = contact[2]  # Store the ID of the wall
                break  # No need to check further if a collision is found

        return collision_info
    
    def check_collision_with_rooms(self):
        collision_info = {
            'has_collided': False,
            'collided_rooms': None
        }

        # Retrieve all contact points
        contact_points = p.getContactPoints(bodyA=self.TURTLE)

        for contact in contact_points:
            # Check if the contact involves the robot and any of the walls
            if contact[2] in self.room_ids:  # contact[2] is the unique ID of the second object in the collision
                collision_info['has_collided'] = True
                collision_info['collided_rooms'] = contact[2]  # Store the ID of the wall
                break  # No need to check further if a collision is found

        return collision_info
    
    def check_collision_with_movable_objects(self):
        collision_info = {
            'has_collided': False,
            'uid': None
        }
        # Retrieve all contact points
        contact_points = p.getContactPoints(bodyA=self.TURTLE)
        for contact in contact_points:
            if contact[2] in self.movable_obj_ids:  # contact[2] is the unique ID of the second object in the collision
                collision_info['has_collided'] = True
                collision_info['uid'] = contact[2]  # Store the ID of the obj
                break  # No need to check further if a collision is found

        return collision_info
    
    def check_collision_with_immovable_objects(self):
        collision_info = {
            'has_collided': False,
            'uid': None
        }
        # Retrieve all contact points
        contact_points = p.getContactPoints(bodyA=self.TURTLE)
        for contact in contact_points:
            if contact[2] in self.immovable_obj_ids:  # contact[2] is the unique ID of the second object in the collision
                collision_info['has_collided'] = True
                collision_info['uid'] = contact[2]  # Store the ID of the obj
                break  # No need to check further if a collision is found

        return collision_info
    
    def check_collision_with_goal_and_update_state(self,agent_position):
        collision_info = {
            'reached_goal': False
        }
        # Retrieve all contact points
        contact_points = p.getContactPoints(bodyA=self.TURTLE)
        for contact in contact_points:
            if contact[2] in [self.goal_id]: 
                collision_info['reached_goal']=True
        
        return collision_info   
    
    def prepare_positional_data_for_obs(self):
        position_data = list(self.robot_position) + list(self.goal_position) + list(self.prev_actions)
        print('[DEBUG] Positional Data : ',np.array(position_data).shape)
        return np.array(position_data)
    
    def prepare_objects_data(self):
        # Create a sorted list of object IDs
        scaled_object_deltas = []
        sorted_object_ids = sorted(self.objectPositions.keys())
        for object_id in sorted_object_ids:
            index = sorted_object_ids.index(object_id)
            object_position = self.objectPositions[object_id]
            # delta = [(object_position[i] - self.robot_position[i])  for i in range(3)]
            object_data = list(object_position) + [self.uid_texture_class_pred[index]] + [self.uid_movable_class_pred[index]]
            scaled_object_deltas.append(object_data)
        print('[DEBUG] : Objets Data : ',np.array(scaled_object_deltas).shape)
        return scaled_object_deltas
            
    def update_uid_texture_class_and_movability(self,sensing_info):
        sorted_object_ids = sorted(self.objectPositions.keys())
        for idx in range(len(sensing_info)):
            info = sensing_info[idx]
            if info['is_facing_object']:
                index = sorted_object_ids.index(info['obj_id'])
                for obj_id, obj in ENV_MANAGER.objects.items():
                    if obj.id == info['obj_id']:
                        if obj.texture_class != None:
                            self.uid_texture_class_pred[index]=obj.texture_class 
                            self.uid_movable_class_pred[index]=obj.casual_probability 
    
    # def calculate_scaled_deltas(self, agent_position):
    #     scaled_goal_delta = []
    #     scaled_object_deltas = []
    #     scaled_walls_deltas = []
    #     scaled_room_deltas = []

    #     # Calculate scaled deltas for walls
    #     for wall_id, wall_midpoint in self.wall_midpoints.items():
    #     # for _wall_id  in self.wall_ids:
    #         # _wall_pos,_ = p.getBasePositionAndOrientation(wall_id)
    #         delta = [((wall_midpoint[i] - agent_position[i])) for i in range(3)]  # 3D delta
    #         scaled_walls_deltas.append(delta)
        
    #     for room_id, room_midpoint in self.room_wall_midpoints.items():  
    #     # for _room_id  in self.room_ids:
    #         # _wall_pos,_ = p.getBasePositionAndOrientation(_room_id)
    #         delta = [((room_midpoint[i] - agent_position[i]) ) for i in range(3)]  # 3D delta
    #         scaled_room_deltas.append(delta)

    #     # Calculate scaled delta for goal
    #     goal_delta = [((self.goal_position[i] - agent_position[i])) for i in range(3)]  # 3D delta

    #     # Create a sorted list of object IDs
    #     sorted_object_ids = sorted(self.objectPositions.keys())
    #     for object_id in sorted_object_ids:
    #         object_position = self.objectPositions[object_id]
    #         delta = [(object_position[i] - agent_position[i])  for i in range(3)]
    #         scaled_object_deltas.append(delta)

    #     return np.array(scaled_goal_delta),np.array(scaled_object_deltas),np.array(scaled_walls_deltas) , np.array(scaled_room_deltas)   
    
    def step(self, action):
        self.prev_actions.append(action)
        self.last_action = action
        self.current_step += 1
        self.render(mode='rgb_array')
        
        self.robot_position,agent_orn = p.getBasePositionAndOrientation(self.TURTLE)
        collision_status = 0

        # Check objects vicinity and Then translate actions
        sensing_info = self.start_sensing_module_and_initializing_digital_mind()
        self.update_moability_in_digital_mind_using_last_action(sensing_info)
        self.casual_reasoning_for_object_movability()
       
        self.previous_position = self.robot_position
        
        self.translate_action(action)
       
        # Update the positions
        self.robot_position,agent_orn = p.getBasePositionAndOrientation(self.TURTLE)
        collision_info = self.check_collision_with_walls()
        if collision_info['has_collided']:
            logging.info('[INFO] Has Colided With Wall ')
            collision_status = 1
            self.reward = -1
            self.done = True  
            
        # Update the positions
        self.robot_position,agent_orn = p.getBasePositionAndOrientation(self.TURTLE)
        collision_info = self.check_collision_with_rooms()
        if collision_info['has_collided']:
            logging.info('[INFO] Has Colided With Rooms ')
            collision_status = 2
            self.reward = 20
            self.done = True 

        obj_collision_info = self.check_collision_with_movable_objects()
        if obj_collision_info['has_collided']:
            self.cummulative_interactions_with_movable_objects += 1
            logging.info(f"[INFO] Has Colided With a Movable object of UID {obj_collision_info['uid']}")
            collision_status = 3
            
        obj_collision_info = self.check_collision_with_immovable_objects()
        if obj_collision_info['has_collided']:
            self.cummulative_interactions_with_immovable_objects += 1
            logging.info(f"[INFO] Has Colided With a Immovable object of UID {obj_collision_info['uid']}")
            collision_status = 4
        
        self.robot_position,agent_orn = p.getBasePositionAndOrientation(self.TURTLE)
        # handle collision with goal - Set a high reward and set done to true
        goal_collision_info = self.check_collision_with_goal_and_update_state(self.robot_position)
        if goal_collision_info['reached_goal']:
            logging.info('[GOAL] Has Reached Goal ')
            logging.info(f'[GOAL] Number of Interactions with Movable Object : {self.cummulative_interactions_with_movable_objects}')
            logging.info(f'[GOAL] Number of Interactions with Immovable Object : {self.cummulative_interactions_with_immovable_objects}')
            logging.ingo(f'[GOAL] Number of Steps Taken to Reach Goal : {self.current_step}')
            collision_status = 5
            self.reward = 200
            self.done = True
        
        # Check if Number of Steps Greater than Max Steps If So Set Episode to be Done - to Prevent the agent to Wander The environment indefinetly during learning
        if self.current_step > self.max_steps:
            logging.info('[INFO] Maximum Steps Reached .. Ending the episode ')
            self.done = True
            
        # Calculate The Cummulative Reward
        self.cumulative_reward = self.reward
        if self.done:
            logging.info(f'[INFO] Episode Ending with Cumlative Reward : {self.cumulative_reward}') # This metric can be used to compare how well the agent performs with and without causal and digital mind
        
        # Compute scaled deltas
        # goal_delta,objects_delta,walls_delta,rooms_delta = self.calculate_scaled_deltas(self.robot_position)
        
        self.update_uid_texture_class_and_movability(sensing_info)
        positional_data = self.prepare_positional_data_for_obs()
        objects_data = self.prepare_objects_data()
        
        observation_space = {
            'positional_data': positional_data,
            'object_data': objects_data,
            'collision_info': collision_status, # 0: No collision, 1: Collided with wall, 2: Collided with movable, 3: Collided with immovable, 4: collided with goal
        }
        
        info = {}
        self.dump_digital_mind_to_json()
        return observation_space,self.reward,self.done,info
    
    def discretize_position(self, position,cell_size=1):
        # Convert a continuous position to a grid cell, assuming position is a tuple (x, y, z)
        grid_x = int(position[0] // cell_size)
        grid_y = int(position[1] // cell_size)
        return grid_x, grid_y
    
    def dump_digital_mind_to_json(self):
        objects_data = {}
        for obj_id, obj in ENV_MANAGER.objects.items():
            # Convert numpy data types to Python native types for JSON serialization
            if obj.texture_class != None:
                objects_data[str(obj_id)] = {
                    'ID': int(obj.id),  # Convert to Python int if it's numpy.int32 or similar
                    'Texture Class': int(obj.texture_class),  # Same conversion
                    'Texture': obj.texture,  # Assuming this is already a serializable type
                    'Current Position': tuple(obj.position),  # Convert to tuple, which should be fine
                    'Initial Position': tuple(obj.initial_position),  # Same as above
                    'Casual Probability': int(obj.casual_probability),  # Convert to Python float if necessary
                    'Movability': bool(obj.movability)  # Ensure it's a native Python boolean
                }
            else:
                objects_data[str(obj_id)] = {
                    'ID': int(obj.id),  # Convert to Python int if it's numpy.int32 or similar
                    'Texture Class':obj.texture_class,  # Same conversion
                    'Texture': obj.texture,  # Assuming this is already a serializable type
                    'Current Position': tuple(obj.position),  # Convert to tuple, which should be fine
                    'Initial Position': tuple(obj.initial_position),  # Same as above
                    'Casual Probability': int(obj.casual_probability),  # Convert to Python float if necessary
                    'Movability': bool(obj.movability)  # Ensure it's a native Python boolean
                }

        file_path = f'{BASE_PATH}/digital_mind_log.json'
        # Write the data to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(objects_data, json_file, indent=4)
            
    def reset(self):
        # Reseting the Sim
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=-90, cameraPitch=-91,
                                            cameraTargetPosition=[9, 9, 0])
        
        # enable real time simulation
        p.setRealTimeSimulation(1)
        for i in range(0, height):
            for j in range(0, width):
                if map_plan[i][j] == "s":
                            startx=i
                            starty=j
        print(f"StartX:{startx}, StartY{starty}")
        self.TURTLE = p.loadURDF(f"{BASE_PATH}/urdf/most_simple_turtle.urdf", [startx, starty, 0])
        self.PLANE = p.loadURDF(f"{BASE_PATH}//urdf/plane_box.urdf")
        self.CRACKED_1 = p.loadTexture(f"{BASE_PATH}/textures/cracked_0052.png")
        self.MARBLED_1 = p.loadTexture(f"{BASE_PATH}/textures/grooved_0051.png")
        self.MARBLED_2 = p.loadTexture(f"{BASE_PATH}/textures/grooved_0048.png")
        
        self.startPos = [0,0,1]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.boxHalfLength = 5.5
        self.boxHalfWidth = 0.5
        self.boxHalfHeight = 3
        self.distance = 100000
        self.last_action = 4 # its in stop action
        
        # Create the Walls 
        self.create_walls(map_plan)
        self.movable_obj_ids = []
        self.immovable_obj_ids = []
        self.rgbImg = None
        self.rgbImgWidth = None
        self.rgbImgHeight = None
        
        # Create the Obstacles
        self.objectIDs = self.create_obstacles(map_plan)# List of unique IDs for each instance
        self.objectPositions = self.initialized_objects_position(self.objectIDs)
        self.goal_position = self.initialized_objects_position(self.goal_id)
        # Set the Goal to be the position of the 3rd Movable Object for now 
        
        # Setup the Agent 
        self.robot_position,self.camera_position,self.robot_orientation = self.setup_agent()
        self.previous_position = self.robot_position
        
        # Reset RL Params
        self.done = False
        self.prev_reward = 0
        self.reward = 0
        goal_delta = distance_3d(self.goal_position,self.robot_position)
        self.visual_predictions= {}
        sorted_obj_ids = [0,1]
        for objectID in sorted_obj_ids:
            self.visual_predictions[objectID] = []
        self.frames=[]
        self.cummulative_interactions_with_movable_objects = 0
        self.cummulative_interactions_with_immovable_objects = 0
        
        # Evaluation Init
        self.current_step = 0
        self.cumulative_reward = 0
        self.goal_reached = False
        self.max_steps = 1000
        
        self.prev_actions = deque(maxlen=AGENT_ACTION_LEN)
        for i in range(AGENT_ACTION_LEN):
            self.prev_actions.append(100) # Creates the History 
        
        # The initial Representation of the environment has to be created here 
        self.start_sensing_module_and_initializing_digital_mind()
        self.casual_reasoning_for_object_movability()
        for obj_id, obj in ENV_MANAGER.objects.items():
            print(f"ObjectStruct(ID={obj.id},Texture Class = {obj.texture_class}, Texture={obj.texture},Current Position={obj.position}, Initial Position={obj.initial_position}, Casual Probability={obj.casual_probability}, Movability={obj.movability})")
        
        logging.info(f'Retained Movability Knowledge : {self.movability_predictions} \n')
        logging.info(f'Retained Movability Predictions : {self.movability_dict}')
        logging.info(f'Retained Visual Prediction Knowledge : {self.visual_predictions} \n')

        # Compute scaled deltas
        # goal_delta,objects_delta,walls_delta,rooms_delta = self.calculate_scaled_deltas(self.robot_position)

        self.uid_texture_class_pred = []
        self.uid_movable_class_pred = []
        for object_id, object_position in self.objectPositions.items(): 
            self.uid_texture_class_pred.append(-1)
            self.uid_movable_class_pred.append(-1)
        
        positional_data = self.prepare_positional_data_for_obs()
        objects_data = self.prepare_objects_data()
        
        observation_space = {
            'positional_data': positional_data,
            'object_data': objects_data,
            'collision_info': 0, # 0: No collision, 1: Collided with wall, 2: Collided with movable, 3: Collided with immovable, 4: collided with goal
        }
       
        return observation_space

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            rgb_array = np.array(self.rgbImg, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (self.rgbImgHeight, self.rgbImgWidth, 4)) # the last dimension contains RGBA values

            # We will discard alpha channel if present
            rgb_array = rgb_array[:, :, :3]

            self.frames.append(rgb_array) # store frames to create video later

            return rgb_array
    
    def close(self):
        if len(self.frames) > 0:
            # Create a video from frames
            with imageio.get_writer(f'{models_dir}episode_video.mp4', fps=20) as video:
                for frame in self.frames:
                    video.append_data(frame)
            self.frames = []  # Reset the frame list for the next episode
        p.disconnect()
    
    def seed(self, seed=None):  
        pass # Not Needed

TIMESTEPS =10000
run = wandb.init(project="[GDP] Search&Rescue-3D", entity="juliangeralddcruz", reinit=True)
wandb.init(
        project="[GDP] Search&Rescue-3D",
        monitor_gym=True,       # automatically upload gym environements' videos
        save_code=True,
    )

env = SearchAndRescueEnv()
env.reset()

model = PPO("MultiInputPolicy", env, verbose=1)
logging.info('[INFO] Learning Started For RL with Causal and Digital Mind')

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
wandb.log({"mean_reward": mean_reward})
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
model.save(f"ppo_{mean_reward}")
env.close()
# Log the video to WandB
video_path = f'{models_dir}episode_video.mp4'  # Assuming env.models_dir is where your videos are saved
wandb.log({"episode_video": wandb.Video(video_path, fps=4, format="mp4")})
run.finish()

# UNIT-TEST
# try:
#     while not done:
#         action = env.action_space.sample()  # Take a random action or implement your control logic here
#         logging.debug(f'Taking a step with action: {action}')
#         observation, reward, done, info = env.step(action)
#         logging.debug(f'Step taken. Observation: {observation}, Reward: {reward}, Done: {done}, Info: {info}')
# except Exception as e:
#     logging.error(f'An error occurred: {e}', exc_info=True)

## Hyperparameter Tuning test 
# def objective(trial):
#     run = wandb.init(project="[GDP] Search&Rescue-3D", entity="juliangeralddcruz", reinit=True)
#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-3)
#     ent_coef = trial.suggest_loguniform('ent_coef', 0.001, 0.1)
#     TIMESTEPS =5000
    
#     config = {
#         "learning_rate":learning_rate,
#         "ent_coef":ent_coef,
#         "timesteps":TIMESTEPS
#     }
    
#     wandb.init(
#         config=config,
#         project="[GDP] Search&Rescue-3D",
#         monitor_gym=True,       # automatically upload gym environements' videos
#         save_code=True,
#     )

#     # TRAINING the RL 
#     models_dir = f"models/{int(time.time())}/"
#     logdir = f"logs/{int(time.time())}/"

#     if not os.path.exists(models_dir):
#         os.makedirs(models_dir)

#     if not os.path.exists(logdir):
#         os.makedirs(logdir)

#     env = SearchAndRescueEnv()
#     env.reset()

#     model = PPO("MultiInputPolicy", env,learning_rate=learning_rate, ent_coef=ent_coef, verbose=1)
#     logging.info('[INFO] Learning Started For RL with Causal and Digital Mind')

#     mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
#     wandb.log({"mean_reward": mean_reward})
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
#     model.save(f"ppo_{mean_reward}")
#     run.finish()
#     return mean_reward

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)
