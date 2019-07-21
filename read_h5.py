import os
import h5py


with h5py.File('./coco_detection.h5') as f:
    print('f:{}\nf.keys:{}'.format(f, f.keys()))
    print(f['dets_labels'][0:10, :, 4])
