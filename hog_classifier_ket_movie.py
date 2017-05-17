import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

import pickle
from lesson_functions import *

# Divide up into cars and notcars
debug_prt=0
model_file='svc_model_cp1.p'

car_images = glob.glob('./vehicles/**/*.png')
cars = []
notcar_images=glob.glob('./non-vehicles/**/*.png')
notcars = []

for image in car_images:
        cars.append(image)

for image in notcar_images:
        notcars.append(image)

if (os.path.isfile(model_file))==True:
    #no need to retrain and extract all hog features
    # just to adjust the scaler
    sample_size=500
    cars=cars[:sample_size]
    notcars=notcars[:sample_size]



if debug_prt:
    print('cars len', len(cars), 'not-cars len', len(notcars))
    print(cars[0])
    print(notcars[0])

### Tweak these parameters and see how the results change.
#color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8
pix_per_cell = 8
#cell_per_block = 2
cell_per_block = 1
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

# pickle the feature parameters
feature_params={'color_space': color_space, 'orient': orient, 'pix_per_cell': pix_per_cell,
                'cell_per_block': cell_per_block, 'hog_channel': "ALL", 'spatial_size':spatial_size,
                'hist_bins': hist_bins, 'spatial_feat':spatial_feat, 'hist_feat': hist_feat,
                'hog_feat': hog_feat}

pickle.dump(feature_params, open('feature_params.p', 'wb'))
# end of pickling

t=time.time()
car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)

notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))


if (os.path.isfile(model_file))==False:
    print('model file is not found, moving to training mode')
    svc = LinearSVC()
    # Check the training time for the SVC

    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    #  Check the score of the SVC
    # print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    #  Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    #pickle the model
    joblib.dump(svc, model_file)
    #joblib.dump(X_scaler, 'X_scaler_model2.p')
    # #pickle.dump(svc, open('svc_model.p', 'wb'))
else:
    print('using the stored model file', model_file)
    svc=joblib.load(model_file)
# technically this is done
#
#
#to see if the model is file is working
    print('Test Accuracy of loaded SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC load predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

img = mpimg.imread('./sample/bbox-example-image.jpg')



def frame_process(img):
    ystart = 440
    ystop = 680
    #scales = [0.3, 0.5, 0.7, 1.0, 1.2, 1.3, 1.5, 2, 2.5]
    scales = [1.1, 1.4, 1.8, 2.4, 2.9, 3.4]
    box_list = []
    for scale in scales:
        out_img, hot_boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                                       hist_bins)
    #transfer hot boxes to the box_list for heat map
        for box_ in hot_boxes:
            box_list.append(box_)

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 18)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img

output_video='P4_ket_out.mp4'

if debug_prt:
    clip1=VideoFileClip("project_video.mp4").subclip(0,2)
else:
    clip1=VideoFileClip("project_video.mp4")

out_clip=clip1.fl_image(frame_process)
out_clip.write_videofile(output_video, audio=False)

print('done')

