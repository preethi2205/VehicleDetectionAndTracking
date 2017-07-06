##Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./output_images/RandomCarImage.png "Random car from training set"
[image2]: ./output_images/RandomNonCarImage.png "Random non-car from training set"
[image3]: ./output_images/RandomCarCSpaceYCrCb.jpg "YCrCb scatter plot - car"
[image4]: ./output_images/RandomCarHogY.png "Y Channel HOG - car"
[image5]: ./output_images/Hog16.png "Hog output with 4 cells"
[image6]: ./output_images/Window64.png "Sliding window 64"
[image7]: ./output_images/Window96.png "Sliding window 96"
[image8]: ./output_images/SlidingWindowEx1.png "Sliding window ex1"
[image9]: ./output_images/SlidingWindowEx2.png "Sliding window ex2"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The function to extract the HOG features is in code cell 6.  

I started by reading in all the `vehicle` and `non-vehicle` images (code cell 7). 

Example of a vehicle image:
![alt text][image1]

Example of a non-vehicle image:
![alt text][image2]

I then explored different color spaces such as HLS, and YCrCb. It was hard to pin point the difference between the pixel distributions for car and non-car images. Here is an example image from the YCrCb exploration of a car image:
 ![alt text][image3]

I explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Y` color channel and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image4]

####2. Explain how you settled on your final choice of HOG parameters.

While choosing HOG parameters, we need to keep in mind that greater the number of pixels per cell, quicker the HOG feature extraction. Going by this principle, I started experimenting with pixel cells of (16,16). For a 64 by 64 image, this would result in 4 cells per image. Here is an example of HOG feature extraction with just 4 cells:
![alt text][image5]

It is clear from the image that the number of pixels per cell of 16 is too large. The HOG features extracted are too generic and do not reveal anything about the shape of the car. Thus, I kept reducing the number of pixels per cell, till I could obtain a HOG image that was somewhat representative of a vehicle. I chose the pixels per cell to be (8,8). 

Increasing the number of cells per block did not cause much difference in the extracted HOG output. Thus, I maintained the cells_per_block parameter at a minimum of 2. I started with the number of orientations to be 3. However, this number was not sufficient to distinguish between different directions in the HOG output. I chose the number of orientations to be 9. Anything greater did not offer too much difference in the HOG feature output. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The functions needed for feature extraction are defined in code cells 3,4,5 and 6.  Code cell 15 sets up the parameters needed for feature extraction. Code cell 6 defines a function that does feature extraction on a list of images (color histogram using YCrCb and HLS, spatial and HOG features using YCrCb images) .  The feature extraction is performed in code cell 16 and 17. The data is then scaled and split into a training and test data set. Code cell 18 sets up a linear SVC and trains the classifier using the training data and labels created.

The classifier runs on 6252 features per image and trained with an accuracy of 99.1% in 23 seconds.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search first begins with the definition of a function called single_img_features which extracts the feature list for a single image (code cell 19).

Another function called slide_windows is defined in code cell 20 to extract a window of a given size (xy_window) from a subset of an image (defined by x_start,y_start to x_stop, y_stop). 

The final function needed for this part is search_windows (code cell 21), which takes the output window from slide_windows() and feature list from single_img_feature. It runs the trained classifier on the feature list and predicts whether the window contains a car. If it does, then a bounding box is added around that window.

Even though these functions achieve the sliding window search, code cell () defines a find_car function, which performs HOG_subsampling. That is, extracting the HOG features once for each search image and sampling a window from the extracted features is more efficient that running HOG feature extraction on each window. The find_car function performs HOG feature extraction on the entire image. It then computes sliding windows, performs sampling to obtain the HOG features for each window. The spatial and histogram features for each window is then extracted and final feature vector is passed to the train classifier to make the prediction. The bounding boxes for the true predictions are then appended to the output list. The findcars function also includes a scaling option to create sliding windows of different sizes. 

 I first used a window size of (64,64) as it is similar to the image size used for HOG feature extraction during the training. However, that resulted in the windows being extremely small:
![alt text][image6]
In order to better fit the cars, I chose a window size of (96,96). Here is an example of the window size (96,96). 
![alt text][image7]
In this project, two window scales are used. The need for multiple windows results from the fact that cars closer to the horizon appear much smaller than the cars close to the vehicle. Thus, a single window size may not capture all cars effectively.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The final set of parameters I used were HOG features on YCrCb channels, 32 histogram bins on the YCrCb image, 32 histogram bins on the HLS image and (16,16) spatial bins. I started out with 16 histogram bins, but had trouble detecting white cars on the road, especially in bright areas. Increasing the number of histogram bins to 32 seemed to train the classifier better. I also added histogram bins from the HLS channels to fix this issue. I searched the image using two scales. However, in order to optimize the performance of the classifier, I restrict the search area of the second scale to Y-values of 350-474 (124 pixels close to the horizon). The scale for this second search is smaller, due to the fact that cars close to the horizon appear small. Here are some examples of the search algorithm on some test images:

![alt text][image8]
![alt text][image9]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected (code cell 25-26).

At this point, I processed the test video and realized that while the detections on the vehicles are pretty accurate, the bounding boxes jump around and are not stable. In order to smooth the detection over time, I average the heat map over the past five frames, and this resulted in much smoother detections.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The following are the steps I took in implementing this project:
1. Analyze different color spaces to see if car images stood out in any. Bright colored cars seem to stand out in saturation. In order to improve the detection of white cars, I add the histogram binning of the HLS color space.
2. Analyze the output of HOG feature extraction in the YCrCb color space. Using a cell size of (8,8) resulted in HOG features that resemble a car.
3. Extract these features for the training set and the test set and train a linear classifier on the training set.
4. Implement a sliding window search algorithm with a window size of (64,64) and run the classifier on each window to find cars within a frame.
5. Perform multi scale window search and average the heat map over the past five frames to obtain a smooth detection on a video pipeline.

Here are things that are likely to fail in the pipeline and some possible fixes:
1. When two cars are close to each other they get detected as one box, as the label function detects them as one blob.
2. There are still some issues detecting white cars on light road surfaces. This could potentially be an issue with the classifier - Data augmentation or using a better classifier would improve this.

In order to continue with the projects, the following are good future steps to take:
1. I would start out by equalizing the brightness and contrast on the training and test set before training the classifier.
2. I would also 
