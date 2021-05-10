# Function to create a 3D volume per group of slices
# Returns normalized numpy arrays holding each slice and associated labels
# CODE CREDITS: Adapted by Jiying Zou from this Kaggle notebook: https://www.kaggle.com/anjum48/reconstructing-3d-volumes-from-metadata

import os
import pydicom
import numpy as np
from utils.window import *

def create_3D_volume(study_df, data_path):
    volume, labels = [], []

    # Iterate over all slices in study (index = slice index, row = slice info)
    for index, row in study_df.iterrows():

        # Read the DCM file
        if row["Dataset"] == "train":
            dcm = pydicom.dcmread(os.path.join(data_path, index + ".dcm"))
        # else:
        #    dcm = pydicom.dcmread(os.path.join(data_path, "stage_1_test_images", index+".dcm"))

        # Get image & 0/1 labels per category
        img = window_img(dcm)
        label = row[["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]]

        volume.append(img)
        labels.append(label)

    volume = np.array(volume)
    labels = np.array(labels)

    # Consolidate labels for each 3D volume -- binary 0/1 of whether any or each of 5 types of ICH present
    labels = (np.sum(labels, axis=0) > 1) - 0

    return volume, labels