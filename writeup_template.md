# **Vehicle Detection Project**

This project involves using opencv and machine learning techniques to detect vehicles on the road. 

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Normalize the features and randomize the dataset for training and testing.
* Implement a sliding-window technique and use my trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/bboxes_and_heat.png
[image5]: ./examples/labels_map.png
[image6]: ./examples/output_bboxes.png
[video7]: ./project_video_result.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how HOG features was extracted from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=7`, `pixels_per_cell=(8, 8)` and `cells_per_block=(1, 1)`:


![alt text][image2]

#### 2. Explain how the final choice of HOG parameters was settled.

The parameters I tested with different color space such as `RGB`, `YUV`, and `YCrCb`. `YCrCb` yields the best performance and accuracy. I also tested a few combinations of orientation, pixels per cell, cells per block. The best accuracy I have achieved is 99.21%.

#### 3. Describe how a classifier is trained using my selected HOG features and color features.

I trained a linear SVM using `LinearSVC`. I define the following functions for feature extraction
- `bin_spatial`: extracts raw pixels
- `color_hist`: extracts color histogram
- `get_hog_features`: extracts hog features

### Sliding Window Search

#### 1. Describe how a sliding window search was implemented.

I decided to search window positions from y=400 to y=700 at different scales. The images of these windows are passed to prediction model to detect cars. I choose xy overlapping that yields the most true positives and least false positives.

#### 2. Show some examples of test images to demonstrate how the pipeline is working, and what I have done to optimize the performance of the classifier.

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]

---

### Video Implementation

#### 1. Provide a link to your final video output.
Here's a [link to my video result](./project_video_result.mp4)


#### 2. Describe how some kind of filter for false positives and some method for combining overlapping bounding boxes were implemented.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here is a frame and its corresponding heatmap:
![alt text][image4]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from a frame:
![alt text][image5]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?  What could you do to make it more robust?

I could have improved prediction accuracy if I had augmented the data. An accuracy of 99.21% is not good enough for practical application on the road. Although I already used tracking method to smooth the boxes, I still encountered some blobs as showed in the video. I have not tested on different environments, driving conditions such as dark night, intensive sun light, snow...etc. 

