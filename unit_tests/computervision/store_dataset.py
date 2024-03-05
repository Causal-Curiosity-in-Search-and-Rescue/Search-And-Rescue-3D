import os
import pandas as pd
import numpy as np

def store_object_information(object_information: dict, save_path: str, csv_file: str = 'texture_dataset.csv'):
    # Check if the CSV file exists in the specified path
    file_path = os.path.join(save_path, csv_file)
    if not os.path.exists(file_path):
        # If the file does not exist, create a new DataFrame and CSV file
        df = pd.DataFrame(columns=[ 'box_type','contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM', 'distance', 'cos_angle'])
    else:
        # If the file exists, load the existing CSV file into a DataFrame
        df = pd.read_csv(file_path)

    # Iterate through each record in the object_information dictionary
    # for record in object_information:
        # Extract features and other information from the record

    # Append the new row to the DataFrame
    df = df.append(object_information, ignore_index=True)

    # Save the updated DataFrame back to CSV
    df.to_csv(file_path, index=False)
