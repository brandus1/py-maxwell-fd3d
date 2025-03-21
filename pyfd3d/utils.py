import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def bwdmean(center_array, w):
    ## Input Parameters
    # center_array: 3D array of values defined at cell centers
    # w: 'x' or 'y' or 'z' direction in which average is taken

    ## Out Parameter
    # avg_array: 2D array of averaged values
    shift = 1
    if(w == 'y'):
        shift = 2 
    if(w == 'z'):
        shift = 3
    
    center_shifted = np.roll(center_array, shift); #doe sthis generalize easily into 3 Dimensions, CHECK!
    avg_array = (center_shifted + center_array) / 2;

    return avg_array

def bwdmean_gpu(center_array, w):
    """
    GPU-accelerated version of bwdmean using CuPy
    
    Parameters:
    -----------
    center_array: CuPy array
        3D array of values defined at cell centers
    w: str
        'x' or 'y' or 'z' direction in which average is taken
        
    Returns:
    --------
    avg_array: CuPy array
        Array of averaged values
    """
    if not GPU_AVAILABLE:
        raise ImportError("CuPy is not available. Cannot use GPU acceleration.")
        
    # Determine axis for roll operation
    axis = 0  # default for 'x'
    if w == 'y':
        axis = 1
    elif w == 'z':
        axis = 2
    
    # Roll the array along the specified axis
    center_shifted = cp.roll(center_array, 1, axis=axis)
    
    # Compute the average
    avg_array = (center_shifted + center_array) / 2
    
    return avg_array