
# **Vehicle Detection Project**


## Kemal Tepe, ketepe@gmail.com

### Ojective: Objective of this work is to utilize machine learning techniques to identify and calissify nearby vehicles in the driving.

### Summary: 

In this project, a linear SVM based classifier is used to identify and clasift surroundung vehicles in video captured during driving conditions. In order to achieve the project objectives, the calssifier is trained by using nearly 16,000 images of cars and non-cars, which is roughly equally distributed. The classifier trained by using image features such as color shpace and histogram oriented gradient (HOG). After obtaining classifier model, the video frames are searched for possible vehicles. In this process, patch of images from the video frames are processed and fed in to the classifier for predictions.If the clasifier identifies a vehicle in a particular path (window), this patch is labed as HOT. The the hot windows are used to geenrate heat map to remove the false positives where classiffy identifies non-vehicle objects and vehicles.  Another method to mitigate the false positives is to use smoothing filter, where we used consecutive heat maps by using  a moving aveage type smoothing filter. The califier yielts roughly 96% accuracy in the test set. The classifier worked well in most of the cases where occasionally identified very close proximity wehicles are one, and some side rails as vehicle. A better training set, such as pictures with more railing can train the model better to eliminate these false positives. Overall, the detection and classification woks well in this problem but enahncement and improvements are possible, as well as optimizations. For example, the feature set can be reduced to speed up the processing without sacrificing the accuracy.

### The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.

* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

* Discuss future enhancements.

### 1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

In order to utilize machine learning (ML) methods, we need a well organized data det. We have obtained this through the resouces section of the project. The data set used in this part of the project includes [GTI vehicle image database]( http://www.gti.ssr.upm.es/data/Vehicle_database.html). The set roughly equal number of vehicle and non-vehicle images of 8,000 images in each bins with 64x64 pixel of each image. Some samples are provided below.

![alt text](./sample/2.png) *non-vehicle*
![alt text](./sample/25.png) *vehicle*


HOG features are extracted from the training images by using functions provided by `skimage.hog()` which was the main function in ```lecture_functions.py``` provided by Udacity, the function API is
```python
features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
```

Tuning and identifying right combination of parameters in to obtain the features were important. I have used the following parameters in my feature extraction:
```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
#cell_per_block = 2
cell_per_block = 1
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
```

**Explain how you settled on your final choice of HOG parameters**.
In the final clasification, `color_space`, and `cell_per_block` parameters were important also including all the channels in `hog_channels`. I experimented with **RGB** color space as well s cell per block of 1 but testing accucary was 2-3% higher with **YCrCb** color space and cell per block of 1. Orientation I kept at 9, 8 worked fine too. My observations were also varified by few blogs that I read such as by [Arnoldo Guzzi](https://chatbotslife.com/vehicle-detection-and-tracking-using-computer-vision-baea4df65906). This blog's author did extensive testing in these parameters. Although he suggests orientation of 8, my classifier worked with orientation of 9 better. 

### 2. Implement a sliding-window technique and use your trained classifier to search for vehicles in images.

**Training classifier**

After extracting HOG features, a linear SVM is trained to obtain the classifier model. The code which generates the classifier is given below:
```python
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
```
Trained model is saved to a file to be used in the future. This also allowed to reduce the testing the images, and video in various runs. Before the training scaling has been done with StandardScaler().

**Sliding Window Search**

The sliding window search was essential to comb the video frames to find the vehicles. The primary function to perform this sliding windows approach is

```python
scales = [1.0, 1.2, 1.4, 1.5, 1.8, 2]
box_list = []
for scale in scales:
  out_img, hot_boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,      spatial_size, hist_bins)
```
```find_cars``` function searches entire img file with 50% overlap with different images scales. Scale of 1 constitutes 64x64 pixel window. I have tried variaty of slaces however scales less than 1 did not help. So I settled on window scales of 
```scales = [1.0, 1.2, 1.4, 1.5, 1.8, 2]```. When sliding window method searches each box for a vehicle and if the window has a vehicle by using classifier, retains this box as hot. After all the scales are searches then the hot boxes are provided to heat map methods to eliminate false positives. The pipeline for the entire provess is defined in ```frame_process``` function. The pipeline of the python code is provided below:

```python
#smoothing
sample_img = mpimg.imread('./sample/bbox-example-image.jpg')
smooth_filter=np.zeros_like(sample_img[:,:,0]).astype(np.float)

def frame_process(img):
    global smooth_filter
    ystart = 400
    ystop = 680
    #scales = [1.0, 1.5, 1.8, 2] #works good
    #below is final
    scales = [1.0, 1.2, 1.4, 1.5, 1.8, 2]
    box_list = []
    heat_map_filter=np.zeros_like(img[:, :, 0]).astype(np.float)
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
    # 7 works good
    #new_heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat=0.3*heat+0.7*smooth_filter
    smooth_filter=heat
    #"{0:.2f}".format(a)
    #heat_thr = smooth_filter.mean() + 5.0 * np.sqrt(smooth_filter.var())
    heat_thr=5
    if debug_prt:
        print('heat: mean, max, var, stdev', "{0:.2f}".format(heat.mean()), "{0:.2f}".format(heat.max()),
              "{0:.2f}".format(np.sqrt(heat.var())),
              'smooth filter: mean, max, var', "{0:.2f}".format(smooth_filter.mean()), "{0:.2f}".format(smooth_filter.max()),
              "{0:.2f}".format(np.sqrt(smooth_filter.var())), "{0:.2f}".format(heat_thr))
    #heat_thr=smooth_filter.mean()+6.0*np.sqrt(smooth_filter.var())
    heat = apply_threshold(heat, heat_thr)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img

```

The pipeline was tested with number of different test images, and below samples  show different stages of the pipeline.

![alt text](./sample/raw_boxes1.png)*Vehicles in the image in raw form where all the hot boxes are shown.*
![alt text](./sample/heat_map1.png)*Vehicles in the image after heat map tresholding to reduce the number of false positives*
![alt text](./sample/final1.png)*Heat map of the hot boxes, which identify concentration of  identification to select vehicles to reduce the false positives.*

Another test image:

![alt text](./sample/raw_boxes2.png)*Vehicles in the image in raw form where all the hot boxes are shown.*
![alt text](./sample/heat_map2.png)*Vehicles in the image after heat map tresholding to reduce the number of false positives*
![alt text](./sample/final2.png)*Heat map of the hot boxes, which identify concentration of  identification to select vehicles to reduce the false positives, notice the false positive at the railing.*



### 3. Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

The pipeline provided in Section 2. is used in processing the video files to identify vehicle in the video frames. However in order to identify vehicles better as well as utilize informatin from frame to frame smoothing filter, technically a moving average filter is applied during obtaining the heat maps. Below the section of this smoothing was provided.

```python
#smoothing
sample_img = mpimg.imread('./sample/bbox-example-image.jpg')
smooth_filter=np.zeros_like(sample_img[:,:,0]).astype(np.float)

def frame_process(img):
    ....
    heat = add_heat(heat, box_list)
    # Apply threshold to help remove false positives
    # 7 works good
    #new_heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat=0.3*heat+0.7*smooth_filter
    smooth_filter=heat
    ....
    return draw_img
```

The smoothing filter is a mvoing average simmilar to **X_k=a*X_current + (1-a)*X_(k-1)**, where a is less than 1.0. The smoothing significantly reduced some of the false positive detections. 

I also tried to adjust the heatmap threshold by using mean and variance of the heat map distribution such as
```
heat_thr = smooth_filter.mean() + 5.0 * np.sqrt(smooth_filter.var())

```

However this did not work as much as I unticipated in this initial trial. That is why a constant threshold value of 5 was selected after few trials. Some enhancements such as using only the searched area and little more elobative method would be used in the future to identify the heatmap threshold. This should further reduce the false positives. 

The output of the pipeline with the project movie clip is given below. There are moments that flase positives as well as false negatives, where the classifier fails to identify vehicles. However these are rare and the classifier achieves a successful identification of the vehicles.

[link to challenge video result](./P5_ket_out_full.mp4) 


### 4. Discussions and Future Work

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Overall, SVM worked well in identifying and clasifying cars and non-cars. The following problems have been faced.

* Although Linear SVM was used, few other methods could be integrated simultaneously to eliminate false positivies. One of the techniques could be tree sombining search mechanism where some obvious image features could be used to eliminate non-cars from the pool.

* The other problem was large number of false postives generated with highway middle raillings, which separetes the directional lanes, and roadsigns. In order to eliminate these false positivies, the classifier could be further trained by images of raillings and road-signs to better model them and identify them in the process.

* The classifier can also be subject more images of vehicles to better train it to reduce false positives. I am not conviced that 8000 images is enough to capture every possible scenario also using only 64x64 pixel images may not be the best set. It is practival but not ideal case.

* Although the project was about SVM classifier, I also wonder (and will probably try) how  a convolutional neural network will clasify the vehicles. We had a good success with road-sign detection. 

* Also better tracking the vehicles from frame to frame also transfering some of the information back to pipeline, similar to what we have used in Advanced Line Detection project would be nise to see. This fill allow us to search boxes identified hot in the previous frame first and adjust the search algorithm to look these areas first and keep a record of these boxes would significantly improve the robustness of the detection and clasifying similar to Markov process.

The project was very educational and interesting. I will keep working on this to fine tune in future. 


