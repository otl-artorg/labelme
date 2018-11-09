import json
import os
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage import io
import numpy as np
import glob

root_dir = '/home/laurent.lejeune/medical-labeling/Dataset25'

im_dir = os.path.join(root_dir, 'input-frames')
im_files = glob.glob(os.path.join(im_dir, '*.jpg'))
im_files = sorted(im_files)

json_dir = os.path.join(root_dir, 'ground_truth-json')
json_files = glob.glob(os.path.join(json_dir, '*.json'))
json_files = sorted(json_files)

out_dir = os.path.join(root_dir, 'ground_truth-frames')

if(not os.path.exists(out_dir)):
    os.mkdir(out_dir)

print("Generating groundtruth masks."\
      "\n Images: {}"\
      "\n Json files: {}\n "\
      "\n Output: {}".format(im_dir,
                             json_dir,
                             out_dir))
for i in range(len(json_files)):
    print('{}/{}'.format(i+1, len(json_files)))
    with open(json_files[i]) as data_file:
        data = json.load(data_file)
    gt_file_out = os.path.join(root_dir,
                               out_dir,
                               os.path.split(im_files[i])[-1])

    im = plt.imread(im_files[i])
    gt = np.zeros((im.shape[0], im.shape[1]))
    for j in range(len(data['shapes'])):
        pts = np.asarray(data['shapes'][j]['points'])
        rr, cc = polygon(pts[:, 1], pts[:, 0])
        gt[rr, cc] = 1
    # print("Saving ground-truth to: " + gt_file_out)
    io.imsave(gt_file_out, gt)
