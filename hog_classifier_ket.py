import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

from lesson_functions import get_hog_features, extract_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()


# Divide up into cars and notcars
debug_prt=1
car_images = glob.glob('./vehicles/**/*.png')
cars = []
notcar_images=glob.glob('./non-vehicles/**/*.png')
notcars = []

for image in car_images:
        cars.append(image)

for image in notcar_images:
        notcars.append(image)

if debug_prt:
    print('cars len', len(cars), 'not-cars len', len(notcars))
    print(cars[0])
    print(notcars[0])

### Tweak these parameters and see how the results change.
colorspace = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 1
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

# pickle the feature parameters
feature_params={'color_space': "RGB", 'orient': orient, 'pix_per_cell': pix_per_cell,
            'cell_per_block': cell_per_block, 'hog_channel': "ALL"}
pickle.dump(feature_params, open('feature_params.p', 'wb'))
# end of pickling

t=time.time()
car_features = extract_features(cars, color_space=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)
notcar_features = extract_features(notcars, color_space=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)
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
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
#pickle 
pickle.dump(svc, open('svc_model.p', 'wb'))