
# import numpy as np
# import gzip

# m = 5

# def extract_labels(filename, num_images):
#     '''
#     Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
#     '''
#     print('Extracting', filename)
#     with gzip.open(filename) as bytestream:
#         print(bytestream)
#         bytestream.read(8)
#         print('bytestreream.read', bytestream.read(8))
#         buf = bytestream.read(1 * num_images)
#         print('buf',buf)
#         labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
#         print(labels)
#     return labels

# y_dash = extract_labels('sample2.tar', m).reshape(m,1)

# print(y_dash)

import gzip, zlib
f = gzip.open('foo.gz', 'wb')
f.write(b"hello world")
f.close()

c = gzip.open("sample2.tar").read(8)
ba = bytearray(c)
print(ba)
buf = gzip.open("sample2.tar").read(1 * 1000)
print(buf)
#x = zlib.decompress(bytes(ba), 15+32)
#print(x)