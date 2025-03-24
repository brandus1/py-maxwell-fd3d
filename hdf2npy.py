from pyhdf.SD import SD, SDC
import numpy as np

def hdf2npy(inputFile):
     
    #open file
    d = SD(inputFile, SDC.READ)

    #data
    data = d.select(0)
    array = data.get()
    
    return array

if __name__ == "__main__":
    import sys
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input HDF file')
    parser.add_argument('output', help='Output NPY file') 
    parser.add_argument('--plot', action='store_true', help='Plot center slices')
    args = parser.parse_args()

    npy = hdf2npy(args.input)
    np.save(args.output, npy)

    if args.plot:
        center = [s//2 for s in npy.shape]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
        
        ax1.imshow(npy[center[0],:,:])
        ax1.set_title('X-Y slice at z={}'.format(center[0]))
        
        ax2.imshow(npy[:,center[1],:])
        ax2.set_title('X-Z slice at y={}'.format(center[1]))
        
        ax3.imshow(npy[:,:,center[2]])
        ax3.set_title('Y-Z slice at x={}'.format(center[2]))
        
        plt.tight_layout()
        plt.show()