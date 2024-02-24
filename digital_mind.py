
"""Digital Mind structure"""

#USAGE:
#env_manager  = EnvironmentObjectsManager()
#Then in a sim loop:
#create_or_update_object(detected_object_id,detected_x,detected_y,detected_z,detected_texture)

class EnvironmentObjectsManager:
    """Environment Objects Manager adds or updates new objects when seen by camera"""

    def __init__(self):
        self.objects = {}  # Dictionary to store objects by ID

    def update_or_create_object(self, object_id,texture_class, **attributes):
        if object_id not in self.objects:
            self.objects[object_id] = ObjectStruct(object_id)
            self.objects[object_id].update_attributes(**attributes,texture_class = texture_class)
        self.objects[object_id].update_attributes(**attributes,texture_class= texture_class)


class ObjectStruct:
    """ Object Struct stores data about each object """
    def __init__(self, object_id):
        self.id = object_id  # ID is now provided as an argument

        # Initialize attributes with default/empty values
        self.texture = None
        self.position = [None,None,None] #{'x': None, 'y': None, 'z': None}  # Using a dictionary to store position
        self.initial_position = [None,None,None] #{'x': None, 'y': None, 'z': None}  # Store the initial position
        self.casual_probability = None
        self.movability = None
        self.texture_class = None

    # Method to update object's attributes
    def update_attributes(self, texture=None, x=None, y=None, z=None, casual_probability=None, movability=None, texture_class = None):

        if self.initial_position[0] is None and x is not None:
            self.initial_position[0] = x
        if self.initial_position[1] is None and y is not None:
            self.initial_position[1] = y
        if self.initial_position[2] is None and z is not None:
            self.initial_position[2] = z

        # Check if the new position differs from the initial position
        if (x is not None and x != self.initial_position[0]) or \
           (y is not None and y != self.initial_position[1]) or \
           (z is not None and z != self.initial_position[2]):
            self.movability = True

        if texture is not None:
            self.texture = texture
        if x is not None:
            self.position[0] = x
        if y is not None:
            self.position[1] = y
        if z is not None:
            self.position[2] = z
        if casual_probability is not None:
            self.casual_probability = casual_probability
        if movability is not None:
            self.movability = movability
        if texture_class is not None:
            self.texture_class = texture_class

    # Representation method for debugging purposes
    def __repr__(self):
        return f"ObjectStruct(ID={self.id},Texture Class = {self.texture_class}, Texture={self.texture}, Position={self.position}, Casual Probability={self.casual_probability}, Movability={self.movability})"