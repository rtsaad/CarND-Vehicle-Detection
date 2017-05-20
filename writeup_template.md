
# Vehicle Detection 

This project consists of a python script to detect vehicles from an image for self-driving purpose. It uses Image Recognition techniques to identify vehicles and draw contour boxes over the vehicles on the original image. The script itself can work over one or a sequence of images (i.e. video).

The main goals of this project is to develop a sequence of operations (pipeline) that gets as input a raw image and outputs an annotate image where the vehicles are marked inside blue boxes. Figures 1 depicts the input and output images.

![alt text][image1]

[//]: # (Image References)
[image1]: ./output_images/output.png
[image2]: ./output_images/examples.png
[image3]: ./output_images/car_feature_spatial_feat.png
[image4]: ./output_images/car_feature_hist_feat.jpg
[image5]: ./output_images/car_feature_hog_feat.jpg
[image6]: ./output_images/car_feature_all_feat.png
[image7]: ./output_images/window_bboxes.png
[image8]: ./output_images/heatmap1.png
[video9]: ./output_images/heatmap2.png


## 1.Access 

The source code for this project is available at [project code]().

## 2.Files

The following files are part of this project:
* window.py:     Camera functions for calibration and perspective transform;
* features.py:   Gradient functions to construct a binary image;
* classifier.py:       Lane functions to keep track, smooth and plot the lane into the video output;
* pipeline.py:   Script to execute the lane detection pipeline of operations;



### Dependency

This project requires the following python packages to work:
* openvc
* numpy
* matplotlib
* moviepy
* pickle

## 3.How to use this project

### Executing the Pipeline

To detect the lane lines, call the following command, together with the file names for the input and output videos.

```sh

python pipeline.py input_video output_video

```

## 4. Project - Introduction


The goals of this project is to detect Vehicles from a video streaming. Therefore, there are fives necessary steps to accomplish this task, they are:

* Load the training data (cars and non cars images);
* Extract the features from the images that best defines the vehicles characteristics for image recognition;
* Normalize, split and randomize the data into training and testing data; 
* Train a classifier to classify images as cars or non cars;
* Implement a Sliding-window, together with the classifier, to search for cars on an image;
* Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Run a pipeline of the tasks described above on a video stream in order to detect and track vehicles. 


## 5. Load Training Data and Feature Vector

This project uses a collection of cars and non cars images gathered from different public sources. The complete set of data contains 17760 images divided into cars  and  non-cars images. Figure 2 presents examples of car and non-cars classes.

![alt text][image2]

After the training data is loaded, the set of selected features are extracted from the image data. The selection of features followed an empirical process taking into account the classifier accuracy and the resulting execution time. Higher number of features results in a better classifier accuracy but also increase the execution time. 

The set of features selected includes: 
* YCbCr color space;
* Spatial binning of size 16x16 pixel resolution;
* Color histogram with 68 bins and range of 0 to 256;
* Histogram of oriented Gradients of the Y channel with 8 orientations and cells of 8x8 pixels;

<!-- 
After the training data is loaded, the set of cars are shuffled before splitting into Train and Test Set. We shuffle the data to try to remove the over fitting caused by the sequence of images when the data is extracted from a video.  
-->

### YCbCr Color Space

Following our empirical process, we have trained our classifier using different color space. For this test, we have used only the HOG feature with only one color channel, testing all three channels and considering only the best result. The YCbCr color space using the Y channel yield the best accuracy among all trials. For the rest of our project we are going to consider the YCbCr color space for all operations regarding image manipulation. 

### Spatial binning

Spatial binning is a technique that reduces the raw image size (all three channels) into a one dimensional feature vector. For our selection, the spatial size of 16x16 pixels yielded the best trade off between classifier accuracy and feature size, which consequently increases the execution time. Figure 2 presents an example of the spatial binning features.

![alt text][image3]

The code for this step is contained at function  'bin_spatial' from the 'feature.py' file (lines 62-66).

### Color histogram

Another approach to identify objects on images is to use the histograms of pixel intensity, also named color histograms, as features. For our project, once again we tried different parameters looking to achieve the best accuracy for our classifier. Our color histogram achieve significant results using 64 number of histograms bins with a range between 0 and 256. Figure 3 shows an example of the color histogram features.

![alt text][image4]

The code for this step is contained at function 'color_hist' from the 'feature.py' file (lines 70-78).

### Histogram of Oriented Gradients (HOG)

The Histograms of Oriented Gradient counts the occurrences of gradient orientation in localized portions of an image. We have used the `skimage.feature,hog()` function with different values until our classifier achieved reasonable accuracy using a non-large feature vector. Using all three channels increases the size of the vector by threefold, which result into almost 5000 features. From our empirical tests, we selected only the first channel, the Y color channel, for our features vector. The use of all channels did not sufficiently improved our classifier to justify the use of such a long vector. From our empirical tests, the rest of the parameters are: we used 8 distinct orientation values, 8 pixels per cell and cell blocks of size 2. Figure 4 shows an example of the HOG features.

![alt text][image5]

It is important to mention that the Histogram of Oriented Gradients features alone proved to be sufficient to achieve reasonable accuracy. The use of the spatial binning and the color histogram are necessary only to reduce the number of false positive. 

The code for this step is contained at function 'get_hog_features' from the 'feature.py' file (lines 42-59).

### Feature Vector

 	
Our final feature vector uses YCbCr color space with 16x16 spatial binning, color histogram with 64 bins and the HOG features from the Y color channel. Figure 5 shows our complete Feature vector, which has 2528 features.

![alt text][image6]

## 6. Classifier

We have tested different classifiers analyzing both the accuracy and the execution time to process the 'project_video' file. Figure 6 presents all three classifier we have tested. The best result was achieved using the SVM with the kernel RBF, which resulted in a smooth video without practically no false positive. However, the execution time is significantly high. 
On the opposite, we have the decision tree classifier that is fast to process but results with a significant number of false positives. Finally, we have the SVM with linear kernel which gives a reasonable result (with just a few false positives) within an acceptable runtime. (For all the videos, we have used a technique of heatmap over several frames to reduce the number of false positives). 

| Classifier    |  Parameter    | Accuracy      | Link Video | Execution Time| False Positive |
|:-------------:|:-------------:|:-------------:|:-------------:| :-------------:|:-------------:|
| SVM / RBF	| C = 10	| 0.9972	| | 66 min | No |
| SVM / LINEAR	| C = 0.1	| 0.9873	| | 27 min | Few|
| Decision Tree	| Min samples= 10, Max depth =	20 | 0.9727 | 11 min | Yes |

From the table above, all three classifier yield reasonable good accuracy, with an slight difference of 1-2% among them. However, the number of false positive among them when processing the videos raises questions. However, it is important to mention that training data comes from video streaming, resulting with a sequence of images where target objects appear almost identical in whole series of images. The problem is that we can not guarantee that our test set does not hold nearly identical images of the train set.

The code for this step is contained at 'classifier.py' file.

## 7. Sliding Window Search

The final step before process the video streaming is to search for target objects on the image. We use the sliding window technique to detect vehicles over the image. 

Our approach uses Hog Sub-sampling window to improve the performance by computing once all the histograms of oriented gradients. We chose the parameter that detects the vehicles with less rate of false positives and also with an acceptable execution time.

Initially, we parameterized our sliding window with a scale of 1, window size of 64 and cell step of 2. Even though the results were optimal (no false positive and high quality detection of vehicles) the execution time was disproportionately long. For instance, rbf took almost three hours to complete.

From the observations of our fist experiment, we realized that vehicles closer are easily detected and do not need such a "good" scale. With that in mind, we implemented a two sliding window search with different scales. We use a scale of 1.2 for the upper part (y coordinated between 400 and 528) and a scale of 1.5 for the bottom part (y coordinated between 500 and 692) of the image. This setup improved the execution time by a third, the rbf implementation took one hour to complete with fairly similar quality result. Figure 7 depicts an example of windows sliding over the image.

![alt text][image7]

The code for this step is contained at function 'find_cars' from 'window.py' file (lines 101-205).

Figure 9 shows an example of a vehicle detection using our complete pipeline (feature vector, window-sliding search, classification and boxes). As we mentioned before at Section Feature Vector, our feature vector holds 2528 features. The selection of this set of features followed an empirical process of trial and error seeking the best result as possible within a reasonable execution time.

![alt text][image1]

-
## 8. Video Implementation

Our best result was achieved using the SVM classifier with 'rbf' kernel. Our pipeline of operations performed reasonably well on the entire project video, detecting vehicles and with an almost negligible error rate (only 1 false positive error). The resulting video is available at this [link](./project_video.mp4).

The code for this step is contained at  'pipeline.py' file.

## 9. False Positive (Heatmap)

We use heatmaps from scipy package (`scipy.ndimage.measurements.label()`) to combine overlapping detections and to remove false positives. Every window the classify detect (possible vehicle) increments its box position into a heatmap. After all box are included into the heatmap, we threshold it in order to remove false positives. Figure 10 and 11 show some examples of overlapping boxes and the resulting heatmap.

![alt text][image8]

![alt text][image9]

Finally, we also have implemented a class to hold the last 15 frames in order to smooth the drawing of boxes and also to remove false positives. This class holds all detected windows for the last 15 frames and use all of them together for the heatmap. Then, the heatmap is threshold with a value higher than the number of frames stored. The code for this class is presented below:


```python
class HeatMap():

    n = 15
    def __init__(self, w, h):
        self.reset(w, h)

    def reset(self, w, h):
        self.box_n = []
        self.heat_w = w
        self.heat_h = h

    def add_heat(self, box_list):
        if len(self.box_n) > self.n:
            self.box_n.pop(0)
        self.box_n.append(box_list)

    def get_heatmap(self):
        heatmap = np.zeros((self.heat_h,self.heat_w)).astype(np.float)        
        # Iterate through list of bboxes
        for box_list in self.box_n:
            for box in box_list:                
                # Add += 1 for all pixels inside each bbox
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1                
            
        # Return thresholded map
        heatmap[heatmap <= len(self.box_n)*(2)] = 0     #1.1 

        return heatmap

``` 

---

## Discussion

During this project, our major concern was to implement a pipeline with a reasonable execution time, i.e. that could be implemented in a real-time system. Since our first implementation, we have managed to significantly reduce  its execution time by considering a smaller feature vector, a smaller number sliding windows and a less complex classifier. However, our pipeline still takes around 30(linear kernel)/60 (rbf kernel) seconds to process 30 frames, or 1 second of video streaming. This is not enough to be used online in a vehicle for self driving purpose. 

We have observed that the number of sliding-windows to be processed for each frame is the main bottleneck of our pipeline. For future work, we are going to study if the translation invariance of Convolution Neural Networks are robust enough to reduce the number of sliding-windows and, by consequence, reduce the execution time. 

