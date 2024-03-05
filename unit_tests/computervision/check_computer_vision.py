import math
import json
import numpy as np
import pybullet as p
import time
import pybullet_data
from digital_mind_objects import EnvironmentObjectsManager
from glcm import get_texture_features,normalise_textures
from maths_functions import calculate_vector,distance_3d,calculate_cos_angle,quaternion_to_forward_vector
from store_dataset import store_object_information
from joblib import load
import pandas as pd 

import pickle
from skimage import io, feature
from sklearn import preprocessing, model_selection, neighbors, metrics
import sklearn
print(sklearn.__version__)

BASE_PATH = "/home/docsun/Documents/git-repo/juliangdz/Search-And-Rescue-3D"

# Load the trained k-NN model and the scaler
model = load(f'{BASE_PATH}/resources/models/unsup_txture_clsf_kmeans.joblib')
scaler = load(f'{BASE_PATH}/resources/models/scaler.joblib')
unsup_scaler = load(f'{BASE_PATH}/resources/models/unsup_txture_clsf_scaler.joblib')



# open the GUI
p.connect(p.GUI)

# load files and place them at the offsets
turtle = p.loadURDF(f"{BASE_PATH}/resources/urdf/most_simple_turtle.urdf", [0, 0, 0])
plane = p.loadURDF(f"{BASE_PATH}/resources/urdf/plane_box.urdf")
#box1 = p.loadURDF("assets/urdf/box.urdf", [1, 0, 0])
#box2 = p.loadURDF("assets/urdf/box.urdf", [1, 1, 0])

# enable real time simulation
p.setRealTimeSimulation(1)

# define gravity
p.setGravity(0, 0, -10)

startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])

#convex mesh from obj
#stoneId = p.createCollisionShape(p.GEOM_MESH, fileName="stone.obj")

boxHalfLength = 5.5
boxHalfWidth = 0.5
boxHalfHeight = 3
segmentLength = 5

text_path_cracked1 = p.loadTexture(f"{BASE_PATH}/resources/textures/cracked_0052.jpg")
text_path_marbled1 = p.loadTexture(f"{BASE_PATH}/resources/textures/grooved_0051.jpg")
text_path_marbled2 = p.loadTexture(f"{BASE_PATH}/resources/textures/grooved_0048.jpg")
wallsIdVisual = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight],
                                    rgbaColor=[0.5, 0.5, 0.5, 1],)

wallsIdCollision = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])

boxIdVisual1 = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[0.5, 0.5, 0.5],
                                  #rgbaColor=[0.5, 1, 0.5, 1],
                                  )
boxIdVisual2 = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[0.5, 0.5, 0.5],
                                  #rgbaColor=[0.5, 1, 0.5, 1],
                                  )
boxIdVisual3 = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[0.5, 0.5, 0.5],
                                  #rgbaColor=[0.5, 1, 0.5, 1],
                                  )
boxIdCollision = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[0.5, 0.5, 0.5])


mass = 1
visualShapeId = -1

p.createMultiBody(baseMass=0,
                    baseVisualShapeIndex=wallsIdVisual,
                    baseCollisionShapeIndex=wallsIdCollision,
                    basePosition=[0, 5, 0],
                  )

p.createMultiBody(baseMass=0,
                  baseVisualShapeIndex=wallsIdVisual,
                  baseCollisionShapeIndex=wallsIdCollision,
                    basePosition=[0, -5, 0])

p.createMultiBody(baseMass=0,
                  baseVisualShapeIndex=wallsIdVisual,
                  baseCollisionShapeIndex=wallsIdCollision,
                    basePosition=[5, 0, 0],
                    baseOrientation=p.getQuaternionFromEuler([0,0,1.5708])
                  )

p.createMultiBody(baseMass=0,
                  baseVisualShapeIndex=wallsIdVisual,
                  baseCollisionShapeIndex=wallsIdCollision,
                    basePosition=[-5, 0, 0],
                    baseOrientation=p.getQuaternionFromEuler([0,0,1.5708])
                  )

box1 =p.createMultiBody(baseMass=1000,
                    baseVisualShapeIndex=boxIdVisual1,
                    baseCollisionShapeIndex=boxIdCollision,
                    basePosition=[0, 2, 1],
                    baseOrientation=p.getQuaternionFromEuler([0,0,0])
                  )
p.changeVisualShape(box1,-1 ,textureUniqueId=text_path_cracked1)

box2 = p.createMultiBody(baseMass=1,
                    baseVisualShapeIndex=boxIdVisual2,
                    baseCollisionShapeIndex=boxIdCollision,
                    basePosition=[2, 0, 1],
                    baseOrientation=p.getQuaternionFromEuler([0,0,0])
                  )
p.changeVisualShape(box2,-1 ,textureUniqueId=text_path_marbled1)

box3 = p.createMultiBody(baseMass=1,
                    baseVisualShapeIndex=boxIdVisual3,
                    baseCollisionShapeIndex=boxIdCollision,
                    basePosition=[-2, 0 , 1],
                    baseOrientation=p.getQuaternionFromEuler([0,0,0])
                  )
p.changeVisualShape(box3,-1 ,textureUniqueId=text_path_marbled2)

distance = 100000
img_w, img_h = 120, 80

# for debug print out the joints of the turtle
for j in range(p.getNumJoints(turtle)):
    print(p.getJointInfo(turtle, j))

forward = 0
turn = 0

######

objectIDs = [box1,box2,box3]# List of unique IDs for each instance
env_manager = EnvironmentObjectsManager()
# Dictionary to store positions for each object ID
objectPositions = {}
for objID in objectIDs:
    pos, orn = p.getBasePositionAndOrientation(objID)
    pos = (int(pos[0]),int(pos[1]),int(pos[2]))
    objectPositions[objID] = pos

# Now, objectPositions dictionary contains the positions of each object
for objID, pos in objectPositions.items():
    print(f"Object ID: {objID}, Position: {pos}")

def create_or_update_object(detected_object_id,texture_class,detected_x,detected_y,detected_z,detected_texture):
    attributes = {
        'texture': detected_texture,
        'x': detected_x,
        'y': detected_y,
        'z': detected_z
    }

    env_manager.update_or_create_object(object_id=detected_object_id,texture_class = texture_class,**attributes)



######
while (1):

    time.sleep(1. / 240.)
    keys = p.getKeyboardEvents()

    leftWheelVelocity = 0
    rightWheelVelocity = 0
    speed = 10

    for k, v in keys.items():

        if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
            turn = -0.5
        if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED)):
            turn = 0
        if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
            turn = 0.5
        if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED)):
            turn = 0

        if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED)):
            forward = 1
        if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED)):
            forward = 0
        if (k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_TRIGGERED)):
            forward = -1
        if (k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED)):
            forward = 0

    rightWheelVelocity += (forward + turn) * speed
    leftWheelVelocity += (forward - turn) * speed

    p.setJointMotorControl2(turtle, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1000)
    p.setJointMotorControl2(turtle, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1000)

    agent_pos, agent_orn = p.getBasePositionAndOrientation(turtle)
    yaw = p.getEulerFromQuaternion(agent_orn)[-1]

    xA, yA, zA = agent_pos
    zA = zA + 0.3 # make the camera a little higher than the robot

    # compute focusing point of the camera
    xB = xA + math.cos(yaw) * distance
    yB = yA + math.sin(yaw) * distance
    zB = zA

    view_matrix = p.computeViewMatrix(
                        cameraEyePosition=[xA, yA, zA],
                        cameraTargetPosition=[xB, yB, zB],
                        cameraUpVector=[0, 0, 1.0]
                    )

    projection_matrix = p.computeProjectionMatrixFOV(
                            fov=90, aspect=1.5, nearVal=0.02, farVal=3.5)

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(img_w, img_h,
                            view_matrix,
                            projection_matrix, shadow=True,
                            renderer=p.ER_BULLET_HARDWARE_OPENGL)


    ###Get Segmented IDs
    img = np.reshape(rgbImg, (height, width, 4)) * 1. / 255.
    gray_scale_image = np.mean(img, axis=2)  # Convert RGB to grayscale by averaging the color channels

    #Get features
    features = get_texture_features(gray_scale_image)
    features = normalise_textures(features,scaler)

    #'segImg' is the segmentation image obtained from 'p.getCameraImage'
    unique_ids = np.unique(segImg)

    # Filter out the ID for the ground plane or any other IDs not relevant
    unique_ids = [uid for uid in unique_ids if uid not in [plane, -1]]

    # Assuming you want to know the positions of these unique IDs
    for uid in unique_ids:
        # Check if uid is within your known objectIDs
        if uid in objectIDs:
            pos, orn = p.getBasePositionAndOrientation(uid)
            print(f"Object ID: {uid}, Position: {pos}")
            pos = (int(pos[0]), int(pos[1]), int(pos[2]))

            # # Convert segImg to a numpy array if it's not already
            # segArray = np.reshape(np.array(segImg), (img_h, img_w))
            # # Create a mask where the segmentation image matches the object ID
            # mask = segArray == uid
            # # Apply the mask to the grayscale image
            # maskedGrayscaleImg = gray_scale_image * mask

            predict_array = np.array([[features['contrast'], features['dissimilarity'], features['homogeneity'],
                                       features['energy'], features['correlation'], features['ASM']]])

            #Get euclidean distance to object
            distance = distance_3d(pos, agent_pos)

            current_visual_prediction = None
            # Calculate the vector to the object

            vector_to_object = calculate_vector(agent_pos, pos)
            # Calculate the cosine of the angle between your orientation and the vector to the object
            cos_angle = calculate_cos_angle(quaternion_to_forward_vector(agent_orn), vector_to_object)
            #print("cos_angle",cos_angle)
            #On getting close to object for inspection, get the predicted class

            if distance<=1 and cos_angle < -0.30: # If im facing
                
                #print("DEBUG:",forward,last_pos_x,agent_pos[0],last_pos_y,agent_pos[1])

                ##Checking if the object is moving by measuring our commands vs position
                if forward == 1 and (((last_pos_x-0.01) < agent_pos[0] < (last_pos_x+0.01)) and ((last_pos_y-0.01) < agent_pos[1] < (last_pos_y+0.01))):
                    #print(f"{uid} NOT MOVING")
                    #Now we make the object we are pushing against's movability = False
                    for obj_id, obj in env_manager.objects.items():
                        if obj.id == uid:
                            # Found the object, now modify its Movability attribute
                            obj.movability = False
                            print(f"Object with ID {uid} has been modified.")

            #Get robot position at last step to aid in classifying movability
            last_pos_x = agent_pos[0]
            last_pos_y = agent_pos[1]
            
            # Updated Vision Texture Classification 
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
            #Use the Model to predict texture class
            scaled_data = unsup_scaler.transform(object_df)
            current_visual_prediction = model.predict(scaled_data)
            print('Current Visual Prediction : ',current_visual_prediction)
            current_visual_prediction +=1 # Makes class 0 = 1, class 1 = 2
            classified_image = "Cracked" if (current_visual_prediction == 2) else "Grooved"
            print(f"The predicted class for the texture is: {classified_image}")

            # For Internal dataset Creation for Vision Training
            object_information = {
                'box_type':uid, # Doesnt make sense to add box_type to dataset 
                'contrast': features['contrast'],
                'dissimilarity': features['dissimilarity'],
                'homogeneity': features['homogeneity'],
                'energy': features['energy'],
                'correlation': features['correlation'],
                'ASM': features['ASM'],
                'distance':distance,
                'cos_angle':cos_angle
            }
            store_object_information(
               object_information,
               save_path='/home/docsun/Documents/git-repo/juliangdz/Search-And-Rescue-3D/unit_tests/computervision/data' 
            )
            
            create_or_update_object(uid,current_visual_prediction, pos[0], pos[1], pos[2],features)

    # print("All Current Objects:")
    # for obj_id, obj in env_manager.objects.items():
    #     print(obj)
    objects_data = {}
    for obj_id, obj in env_manager.objects.items():
        # Convert numpy data types to Python native types for JSON serialization
        objects_data[str(obj_id)] = {
            'ID': str(obj.id),  # Convert to Python int if it's numpy.int32 or similar
            'Texture Class': str(obj.texture_class),  # Same conversion
            'Texture': obj.texture,  # Assuming this is already a serializable type
            'Current Position': tuple(obj.position),  # Convert to tuple, which should be fine
            'Initial Position': tuple(obj.initial_position),  # Same as above
            'Casual Probability': str(obj.casual_probability),  # Convert to Python float if necessary
            'Movability': bool(obj.movability)  # Ensure it's a native Python boolean
        }
        
        

    file_path = f'digital_mind_log.json'
    # Write the data to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(objects_data, json_file, indent=4)