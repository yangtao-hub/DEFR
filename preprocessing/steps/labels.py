import numpy as np

def binarize(array):
    return (array > 0).astype(np.uint8)
