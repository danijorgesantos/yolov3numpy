
import numpy as np
import gzip

from PIL import Image

m = 5

def extract_labels(filename, num_images):
    '''
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        print(bytestream)
        bytestream.read(8)
        print('bytestreream.read', bytestream.read(8))
        buf = bytestream.read(1 * num_images)
        print('buf',buf)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        print(labels)
    return labels

y_dash = extract_labels('sample2.tar', m).reshape(m,1)