import json
import os
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage import io
import numpy as np
import glob

the_dir = '/home/krakapwa/Desktop/slitlamps/3/input-frames/'
im_files = glob.glob(os.path.join(the_dir, '*.jpg'))
im_files = sorted(im_files)
data_files = glob.glob(os.path.join(the_dir, '*.lif'))
data_files = sorted(data_files)

for i in range(len(data_files)):
    with open(data_files[i]) as data_file:
        data = json.load(data_file)
    gt_file_out = os.path.splitext(im_files[i])[0] + '_gt.png'

    im = plt.imread(im_files[i])
    gt = np.zeros((im.shape[0], im.shape[1]))
    for j in range(len(data['shapes'])):
        pts = np.asarray(data['shapes'][j]['points'])
        rr, cc = polygon(pts[:, 1], pts[:, 0])
        gt[rr, cc] = 1
    print("Saving ground-truth to: " + gt_file_out)
    io.imsave(gt_file_out, gt)
