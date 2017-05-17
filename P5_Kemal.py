import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from lesson_functions import *

debug_prt=1

# how we pickled while obtaining classifier model feature parameters
'''
feature_params={'color_space': "RGB", 'orient': orient, 'pix_per_cell': pix_per_cell,
                'cell_per_block': cell_per_block, 'hog_channel': "ALL", 'spatial_size':spatial_size,
                'hist_bins': hist_bins, 'spatial_feat':spatial_feat, 'hist_feat': hist_feat,
                'hog_feat': hog_feat}
'''
#
feature_params= pickle.load(open("feature_params2.p", "rb"))
color_space=feature_params["color_space"]
orient = feature_params["orient"]
pix_per_cell = feature_params["pix_per_cell"]
cell_per_block = feature_params["cell_per_block"]
hog_channel=feature_params["hog_channel"]
spatial_size = feature_params["spatial_size"]
hist_bins = feature_params["hist_bins"]
spatial_feat=feature_params["spatial_feat"]
hist_feat=feature_params["hist_feat"]
hog_feat=feature_params["hog_feat"]
#
#parameters are obtained
img = mpimg.imread('./sample/bbox-example-image.jpg')


ystart = 400
ystop = 720
scale = 1.5

# Fit a per-column scaler
X_scaler = StandardScaler()


#out_img = find_cars(img, ystart, ystop, scale, svc2, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
#                    hist_bins)

#out_img = find_cars(img, ystart, ystop, scale, svc2,  orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

plt.imsave('./out_sample/p5_out_img.png',out_img)
