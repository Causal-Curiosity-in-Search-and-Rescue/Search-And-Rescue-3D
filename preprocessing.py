from skimage.feature import graycomatrix, graycoprops
import numpy as np

def get_texture_features(image):
    GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    # Assume image is grayscale, normalize to range 0-255, and convert to uint8
    image_uint8 = (image / image.max() * 255).astype('uint8')
    # Compute GLCM and its properties
    glcm = graycomatrix(image_uint8, [1], [0], 256, symmetric=True, normed=True)
    features = {prop: graycoprops(glcm, prop)[0, 0] for prop in GLCM_PROPS}
    features["contrast"] = features["contrast"]/100
    return features

def normalise_textures(features,scaler):
    # Create an instance of MinMaxScaler


    # Convert the features dictionary to a list of values and then to a 2D array
    feature_values = np.array(list(features.values())).reshape(-1, 1)

    # Fit the scaler to the feature values and transform
    normalized_feature_values = scaler.fit_transform(feature_values).flatten()

    # Create a new dictionary with the normalized values
    normalized_features = dict(zip(features.keys(), normalized_feature_values))

    return normalized_features
