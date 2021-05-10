# Function to get the metadata of all the DCM files given the directory
# CODE CREDITS: https://www.kaggle.com/anjum48/reconstructing-3d-volumes-from-metadata
import os
import pydicom
import pandas as pd
from tqdm import tqdm_notebook

def get_metadata(image_dir):
    # Set up the metadata dictionary for all images
    labels = [
        'BitsAllocated', 'BitsStored', 'Columns', 'HighBit',
        'ImageOrientationPatient_0', 'ImageOrientationPatient_1', 'ImageOrientationPatient_2',
        'ImageOrientationPatient_3', 'ImageOrientationPatient_4', 'ImageOrientationPatient_5',
        'ImagePositionPatient_0', 'ImagePositionPatient_1', 'ImagePositionPatient_2',
        'Modality', 'PatientID', 'PhotometricInterpretation', 'PixelRepresentation',
        'PixelSpacing_0', 'PixelSpacing_1', 'RescaleIntercept', 'RescaleSlope', 'Rows', 'SOPInstanceUID',
        'SamplesPerPixel', 'SeriesInstanceUID', 'StudyID', 'StudyInstanceUID',
        'WindowCenter', 'WindowWidth', 'Image',
    ]

    data = {l: [] for l in labels}  # like {'BitsAllocated': [], 'BitsStored': [], etc.}

    # Update the metadata for all DCM files
    for image in tqdm_notebook(os.listdir(image_dir)):  # for each image

        name, extension = os.path.splitext(image)  # check if file is a DCM file (addded by JZ)
        if extension == '.dcm':  # if it is, then:

            data["Image"].append(image[:-4])  # append image id
            ds = pydicom.dcmread(os.path.join(image_dir, image))  # read the whole DCM file

            # update metadata
            for metadata in ds.dir():
                if metadata != "PixelData":
                    metadata_values = getattr(ds, metadata)  # get value of each metadata param
                    if type(metadata_values) == pydicom.multival.MultiValue and metadata not in ["WindowCenter",
                                                                                                 "WindowWidth"]:
                        for i, v in enumerate(metadata_values):
                            data[f"{metadata}_{i}"].append(v)  # update metadata value in dict
                    else:
                        if type(metadata_values) == pydicom.multival.MultiValue and metadata in ["WindowCenter",
                                                                                                 "WindowWidth"]:
                            data[metadata].append(metadata_values[0])
                        else:
                            data[metadata].append(metadata_values)

    return pd.DataFrame(data).set_index("Image")