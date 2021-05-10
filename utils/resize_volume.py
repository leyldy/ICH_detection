# Function to resize an image via spline interpolation
# CODE CREDITS: https://keras.io/examples/vision/3D_image_classification/#:~:text=A%203D%20CNN%20is%20simply,learning%20representations%20for%20volumetric%20data
from scipy import ndimage

def resize_volume(img, d, w, h):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = d
    desired_width = w #512
    desired_height = h #512
    # Get current depth
    current_depth = img.shape[0]
    current_width = img.shape[1]
    current_height = img.shape[2]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate -- don't think we need to do this...?
    #img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1) # linear spline interpolation
    return img