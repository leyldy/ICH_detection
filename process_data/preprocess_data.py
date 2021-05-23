'''
Primary author: Jiying Zou
Purpose: Take subsetted CT scans (from subsample_raw_data.py) and create ordered, interpolated 3D volumes for scans with 30-50 slices
Notes: This script is meant to be run after subsample_raw_data.py (which itself should only be run once)
Code credits: Lines 29-67 adapted from a Kaggle notebook https://www.kaggle.com/anjum48/reconstructing-3d-volumes-from-metadata
              Code credits for functions used are separately specified in the function files.
'''

import pandas as pd
import numpy as np
from scipy import ndimage
import os
#import matplotlib.pylab as plt
#%matplotlib inline

# import pydicom
# from tqdm import tqdm_notebook
# import pickle
import random
import sys

sys.path.append('..')

from utils.get_metadata import *
#from utils.window import window_img
from utils.create_3D_volume import *
from utils.resize_volume import *

random.seed(231)


# Set file paths
data_path = '/Users/zouj6/Documents/CS231N/final_proj/stage_2_train/'
metadata_path = data_path
label_path = '/Users/zouj6/Documents/CS231N/final_proj/stage_2_train.csv'

# Prepare labels and metadata (for each image, is ICH present? if so, what type?)
train_df = pd.read_csv(f'{label_path}').drop_duplicates()
train_df['ImageID'] = train_df['ID'].str.slice(stop=12)
train_df['Diagnosis'] = train_df['ID'].str.slice(start=13)
train_labels = train_df.pivot(index="ImageID", columns="Diagnosis", values="Label")

# Generate metadata dataframe
train_metadata = get_metadata(data_path)
train_metadata["Dataset"] = "train"

# Save metadata dataframe (only have to do this once since it takes a while, results are saved)
train_metadata.to_parquet(f'{data_path}/train_metadata_samp.parquet.gzip', compression='gzip')

# Load metadata dataframe
#train_metadata = pd.read_parquet(f'{metadata_path}/train_metadata_samp.parquet.gzip')

# Join image metadata with outcome labels
metadata = train_metadata.join(train_labels)

# Organize the scans (top-down order)
metadata.sort_values(by="ImagePositionPatient_2", inplace=True, ascending=False) # sort so images are in top-down order

# Count the number of CT scans
print("There are ", metadata["StudyInstanceUID"].nunique(), " CT scans in the subsampled dataset.")

# Group together all images per scan
studies = metadata.groupby("StudyInstanceUID")
studies_list = list(studies)

## Descriptive statistics on num slices per scan
print("Distribution of number of slices per scan:")
print(studies.size().describe())

## Visualization for num slices per scan
#plt.hist(studies.size())

# Subset for just the scans with 30-50 slices (histogram-driven cutoff)
# (Some clinical-practice considerations for num images per scan can be found here: https://www.reddit.com/r/askscience/comments/7cruuv/how_many_images_does_a_ct_scan_produce/)
studies_list_mod = [study for study in studies_list if study[1].shape[0] in range(30, 51)]

# How many unique studies (CT scans) are we dealing with?
# print("After subsetting for CT scans with 30-50 slices, we are left with ", len(set(studies_list_mod)), " unique scans.")


# Create 3D volumes for all the scans
# (Also does data NORMALIZATION and L/R FLIPS with probability 0.5, and keeps track of largest dimensions for interpolation later on)
# (Not doing rotations or up-down flips or crops since CT scans are usually all front side up and full-brain-cross-section anyways)
study_names_all = []
volumes_all = []
labels_all = []

largest_depth = 0
largest_width = 0
largest_height = 0

flip_prob = 0.5

print("Creating 3D volumes.........")
for i in range(len(studies_list_mod)):
  if i % 100 == 0:
      print("On scan #" + str(i+1))

  # Remove this after checking code works
  #if i == 102:
  #    break

  # Get study
  study_name, study_df = studies_list_mod[i]

  # Create 3D volume (normalizes)
  volume, labels = create_3D_volume(study_df, data_path) # label returned is just a length=6 list containing 0/1 for any ICH / ICH type

  # L/R flip with prob = 0.5
  if np.all(np.random.uniform(0, 1, size = 1)) > flip_prob:
      volume = volume[:, :, ::-1]

  # Store results
  study_names_all.append(study_name)
  volumes_all.append(volume)
  labels_all.append(labels)

  # Update largest depth, width, height
  shape = volume.shape
  if shape[0] > largest_depth:
    largest_depth = shape[0]
  if shape[1] > largest_width:
    largest_width = shape[1]
  if shape[2] > largest_height:
    largest_height = shape[2]

print("DONE!")
print("Largest depth, width, height are: " + str(largest_depth) + ", " + str(largest_width) + ", " + str(largest_height))


# First-order spline interpolation (so that data is all of same size)
print("Adjusting all scans to be of same size.............")

volumes_all_interp = []
for i in range(len(volumes_all)):
  if i % 100 == 0:
    print("On scan #" + str(i+1))

  resized_volume = resize_volume(volumes_all[i], largest_depth, largest_width, largest_height)
  volumes_all_interp.append(resized_volume)

print("DONE!")




# SAVE RESULTS ------------------------------------------------------------------------------------
'''
with open('CTscans_studynames_train', 'wb') as studies_file:
    pickle.dump(study_names_all, studies_file)

with open('CTscans_3Dvolumes_train', 'wb') as volume_file:
    pickle.dump(volumes_all_interp, volume_file)

with open('CTscans_3Dlabels_train', 'wb') as labels_file:
    pickle.dump(labels_all, labels_file)













########################
### GET DATA SAMPLES ###
########################

# MERGE CT SCAN SLICE METADATA WITH OUTCOME LABELS ------------------------------------------------
# CODE CREDS: Adapted from this Kaggle notebook https://www.kaggle.com/anjum48/reconstructing-3d-volumes-from-metadata
# Note: The CSV file contains only two columns: one with the slice (image) ID and ICH type,
#       and another for whether that type of ICH exists in the slice. We need to first break
#       this info apart and pivot it so that we get indicators for whether each slice has ICH type.
#       The DCM files contain slice metadata in addition to the 2D slice pixel array itself.
#       We need to extract the metadata and combine it with outcome labels to get the whole picture
#       per slice.

print("Combining metadata with outcomes............")

# 1. Prepare outcome labels per slice
#    [slice ID | any ICH | ICH type flags]
#    This step drops duplicates
train_df = pd.read_csv(label_path_train).drop_duplicates()
train_df['ImageID'] = train_df['ID'].str.slice(stop=12)
train_df['Diagnosis'] = train_df['ID'].str.slice(start=13)
train_labels = train_df.pivot(index="ImageID", columns="Diagnosis", values="Label")


# 2. Prepare metadata per slice
# Generate metadata dataframes and export as zipped file
# (only have to do this once, since it takes a while)
train_metadata = get_metadata(data_path_train)
train_metadata.to_parquet(data_path_train + "/train_metadata.parquet.gzip", compression='gzip')

#test_metadata = get_metadata(os.path.join(data_path_test, "test_images"))
#test_metadata.to_parquet(f'{data_path_test}/test_metadata.parquet.gzip', compression='gzip')


# 3. Join slice metadata with outcomes
train_metadata = pd.read_parquet(metadata_path_train + "/train_metadata.parquet.gzip")
train_metadata["Dataset"] = "train"
train_metadata = train_metadata.join(train_labels)
metadata = train_metadata

#test_metadata = pd.read_parquet(f'{metadata_path_test}/test_metadata.parquet.gzip')
#test_metadata["Dataset"] = "test"

#metadata = pd.concat([train_metadata, test_metadata], sort=True)

# 4. Organize the scans
metadata.sort_values(by="ImagePositionPatient_2", inplace=True, ascending=False) # sort so images are in top-down order

# Num unique studies (scans)
print("There are ", metadata["StudyInstanceUID"].nunique(), " CT scans in the whole dataset.")


# SUB-SAMPLE TRAINING SET -------------------------------------------------------------------------
# CODE CREDS: Jiying wrote this part

print("Sub-sampling the data...........")

# Specify sub-sample sizes per outcome class
n_normal = 750
n_ICH_epidural = n_normal // 5
n_ICH_intraparenchymal = n_normal // 5
n_ICH_intraventricular = n_normal // 5
n_ICH_subarachnoid = n_normal // 5
n_ICH_subdural = n_normal // 5

# Get unique study IDs for sub-samples
outcome_types = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
outcome_sampsizes = [n_normal, n_ICH_epidural, n_ICH_intraparenchymal, n_ICH_intraventricular, n_ICH_subarachnoid,
                     n_ICH_subdural]
study_outcomes = metadata[["StudyInstanceUID"] + outcome_types].drop_duplicates()

studyIDs = []
for status, n in zip(outcome_types, outcome_sampsizes):
    # Sample some number of unique scans per outcome
    if status == "any":  # normal
        new_studyIDs = list(
            np.random.choice(study_outcomes[study_outcomes[status] == 0]["StudyInstanceUID"].unique(), n))
    else:  # not normal (ICH)
        new_studyIDs = list(
            np.random.choice(study_outcomes[study_outcomes[status] == 1]["StudyInstanceUID"].unique(), n))

    studyIDs += new_studyIDs

# Subsample the scans (subset the dataframe)
metadata = metadata[metadata['StudyInstanceUID'].isin(studyIDs)]


# TURN CT SCANS INTO 3D VOLUMES -------------------------------------------------------------------
# CODE CREDS: Adapted from this Kaggle notebook https://www.kaggle.com/anjum48/reconstructing-3d-volumes-from-metadata
#             with windowing function taken from this notebook: https://www.kaggle.com/wfwiggins203/eda-dicom-tags-windowing-head-cts
#             and interpolation adapted from this guide: https://keras.io/examples/vision/3D_image_classification/#:~:text=A%203D%20CNN%20is%20simply,learning%20representations%20for%20volumetric%20data
# Note: The output should be one 3D volume (numpy array) per CT scan, with all scans being interpolated to be the same size.
#       Values are normalized.

# Group together all images per scan (scans were organized in top-down order already earlier)
studies = metadata.groupby("StudyInstanceUID")

print("Distribution of number of slices per scan:")
print(studies.size().describe())

studies = list(studies) # a list of CT scan ID, then the metadata per slice

del metadata # Python will delete if no longer referenced

# Create 3D volumes for all the scans (does normalization too)
# Keep track of largest dimensions for interpolation later on
#(Jiying wrote this part)
study_names_all = []
volumes_all = []
labels_all = []

largest_depth = 0
largest_width = 0
largest_height = 0

print("Creating 3D volumes.........")
for i in range(len(studies)):
  if i % 100 == 0:
    print("On scan #" + str(i+1))

  # Get study
  study_name, study_df = studies[i]

  # Create 3D volume
  volume, labels = create_3D_volume(study_df, data_path_train) # label returned is just a length=6 list containing 0/1 for any ICH / ICH type

  # Store results
  study_names_all.append(study_name)
  volumes_all.append(volume)
  labels_all.append(labels)

  # Update largest depth, width, height
  shape = volume.shape
  if shape[0] > largest_depth:
    largest_depth = shape[0]
  if shape[1] > largest_width:
    largest_width = shape[1]
  if shape[2] > largest_height:
    largest_height = shape[2]

print("DONE!")
print("Largest depth, width, height are: " + str(largest_depth) + ", " + str(largest_width) + ", " + str(largest_height))


# First-order spline interpolation (so that data is all of same size)
print("Adjusting all scans to be of same size.............")

volumes_all_interp = []
for i in range(len(volumes_all)):
  if i % 100 == 0:
    print("On scan #" + str(i+1))

  resized_volume = resize_volume(volumes_all[i], largest_depth, largest_width, largest_height)
  volumes_all_interp.append(resized_volume)

print("DONE!")

del volumes_all # no longer necessary


# SAVE RESULTS ------------------------------------------------------------------------------------
with open('CTscans_studynames_train', 'wb') as studies_file:
    pickle.dump(study_names_all, studies_file)

with open('CTscans_3Dvolumes_train', 'wb') as volume_file:
    pickle.dump(volumes_all_interp, volume_file)

with open('CTscans_3Dlabels_train', 'wb') as labels_file:
    pickle.dump(labels_all, labels_file)
'''