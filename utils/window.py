# This file contains functions to help process the CT scans
# CODE CREDS: https://www.kaggle.com/wfwiggins203/eda-dicom-tags-windowing-head-cts
import pydicom
import numpy as np

# Windowing function (the BW filter)
def window_img(dcm, width=None, level=None, norm=True):
    pixels = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

    # Pad non-square images
    if pixels.shape[0] != pixels.shape[1]:
        (a, b) = pixels.shape
        if a > b:
            padding = ((0, 0), ((a - b) // 2, (a - b) // 2))
        else:
            padding = (((b - a) // 2, (b - a) // 2), (0, 0))
        pixels = np.pad(pixels, padding, mode='constant', constant_values=0)

    if not width:
        width = dcm.WindowWidth
        if type(width) != pydicom.valuerep.DSfloat:
            width = width[0]
    if not level:
        level = dcm.WindowCenter  # center
        if type(level) != pydicom.valuerep.DSfloat:
            level = level[0]
    lower = level - (width / 2)  # lower boundary
    upper = level + (width / 2)  # upper boundary
    img = np.clip(pixels, lower, upper)

    if norm:
        return (img - lower) / (upper - lower)  # normalization
    else:
        return img