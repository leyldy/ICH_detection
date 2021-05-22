'''
Primary author: Jiying Zou
Purpose: The full training data is ~500 GB downloaded from Kaggle. The following code is to help subsample approx 4000 CT scans
Code credits: Lines 19-47 adapted from a Kaggle notebook https://www.kaggle.com/anjum48/reconstructing-3d-volumes-from-metadata
'''



# !pip install pydicom
# !pip install pyarrow
import pandas as pd
import numpy as np
import random
import os
import pydicom
import pyarrow
from tqdm import tqdm_notebook
from utils.get_metadata import *

# Set file paths
data_path = '/Volumes/Seagate Backup Plus Drive/rsna-intracranial-hemorrhage-detection/stage_2_train/'
metadata_path = data_path
label_path = '/Volumes/Seagate Backup Plus Drive/rsna-intracranial-hemorrhage-detection/stage_2_train.csv'


# Prepare labels and metadata (for each image, is ICH present? if so, what type?)
train_df = pd.read_csv(f'{label_path}').drop_duplicates()
train_df['ImageID'] = train_df['ID'].str.slice(stop=12)
train_df['Diagnosis'] = train_df['ID'].str.slice(start=13)
train_labels = train_df.pivot(index="ImageID", columns="Diagnosis", values="Label")

# Generate metadata dataframe
train_metadata = get_metadata(data_path)

# Save metadata dataframe (only have to do this once since it takes a while, results are saved)
train_metadata.to_parquet(f'{data_path}/train_metadata.parquet.gzip', compression='gzip')

# Load metadata dataframe
train_metadata = pd.read_parquet(f'{metadata_path}/train_metadata.parquet.gzip')

# Join image metadata with outcome labels
metadata = train_metadata.join(train_labels)

# Organize the scans (so scans go from top to bottom)
metadata.sort_values(by="ImagePositionPatient_2", inplace=True, ascending=False) # sort so images are in top-down order

# Count the number of CT scans
print("There are ", metadata["StudyInstanceUID"].nunique(), " CT scans in the whole dataset.")

# Subsample the scans and get relevant slices (study IDs)
random.seed(3)

n_normal = 2000
n_ICH_epidural = n_normal // 5
n_ICH_intraparenchymal = n_normal // 5
n_ICH_intraventricular = n_normal // 5
n_ICH_subarachnoid = n_normal // 5
n_ICH_subdural = n_normal // 5

outcome_types = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
outcome_sampsizes = [n_normal, n_ICH_epidural, n_ICH_intraparenchymal, n_ICH_intraventricular, n_ICH_subarachnoid,
                     n_ICH_subdural]
study_outcomes = metadata[["StudyInstanceUID"] + outcome_types].drop_duplicates()

studyIDs = []
for status, n in zip(outcome_types, outcome_sampsizes):
    # Sample some number of unique scans per outcome
    if status == "any":  # normal
        new_studyIDs = list(np.unique(np.random.choice(study_outcomes[study_outcomes[status] == 0]["StudyInstanceUID"].unique(), n)))
    else:  # not normal (ICH)
        new_studyIDs = list(np.unique(np.random.choice(study_outcomes[study_outcomes[status] == 1]["StudyInstanceUID"].unique(), n)))

    studyIDs += new_studyIDs
    print("Number of " + status + " scans: " + str(len(new_studyIDs)))

# Subsample the scans (subset the dataframe)
metadata = metadata[metadata['StudyInstanceUID'].isin(studyIDs)]

# Extract the image IDs relevant to our sample of scans
imageIDs = list(metadata.index)
for file_idx in range(len(imageIDs)):
    imageIDs[file_idx] += '.dcm'

# Keep only those images in the file folder (THIS IS DANGEROUS -- RUN ONLY ONCE)
# (These commands will rename files that we want to delete, then go to terminal and use a command line prompt to delete such files)
os.chdir(data_path)
var = 'del-'
for filename in os.listdir(data_path):
    if filename not in imageIDs:
        os.rename(filename, var+filename)

# This can also be run from terminal -- just cd into the /stage_2_train/ folder
# !find . -name "del-*" -type f -delete
