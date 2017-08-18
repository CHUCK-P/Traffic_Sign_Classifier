#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/plot1.png "Dataset Class Visualization Original"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/test_01_32x32_speed_limit_30kpm.png "Speed Limit 30kpm"
[image5]: ./examples/test_04_32x32_speed_limit_70kpm.png "Speed Limit 70kpm"
[image6]: ./examples/test_11_32x32_row_next_intersection.png "Right of Way Next Intersection"
[image7]: ./examples/test_25_32x32_road_work.png "Road Work"
[image8]: ./examples/test_33_32x32_turn_right_ahead.png "Turn Right Ahead"
[image9]: ./examples/plot2.png "Dataset Class Visualization Augmented"
[image10]: ./examples/plot3.png "Training Loss and Validation Loss"
[image11]: ./examples/lenet.png "LeNet Model"

Thanks for reading it! Here is a link to my [project code](https://github.com/CHUCK-P/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 examples
* The size of the validation set is 4410 examples
* The size of test set is 12630 examples
* The shape of a traffic sign image is 32 x 32 x 3 (pixels x pixels x color channels)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of number of samples per unique class

![alt text][image1]

###Design and Test a Model Architecture

###Preprocessing Steps

I created a pipeline to be reused on every image that was for training, validation, testing, and supplementary testing.  A summary of the steps that I used are as follows:
1) Convert to grayscale
2) Normalize the image between -1.0 to 1.0
3) Crop the image
4) Sharpen the image

As a first step, I decided to convert the images to grayscale because lighting conditions may affect the different channels in RGB, adversely.  I don't have any experimentation to support that assumption, though.  Just through experience on other projects.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Next, I normalized the image data to between -1.0 and 1.0 to ensure that the data was properly conditioned for gradient descent

Then, I cropped the image to eliminate unnecessary features that might cause confuse the model

Finally, I sharpened the image using a Gaussian blur overlayed with the original image in order to reduce noise that are not relevant features

After visualizing the dataset via the histogram plot, I decided to generate additional data because a siginificant number of the traffic sign classes had less than 1200 samples available.

To add more data to the the data set, I sought out classes with less than 1200 samples and then used a rotation transform.  The additional data was simply appended to the original data set.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 

![alt text][image1]

![alt text][image9]

####2. MODEL ARCHITECTURE
My model architecture is heavily based upon the LeNet example presented for the MNIST example.  The only small modifications that I made were to add dropout after the first and second RELU activations.

![alt text][image11]

My final model consisted of the following layers:

| Layer         		|     Description	        			                 		| 
|:---------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					             		| 
| Convolution 5x5 | 1x1 stride, same padding, outputs 28x28x6    	|
| RELU					       |												|
| Dropout 1       | Used for training ONLY                        |
| Pooling         | Output 14x14x6                                |
| Convolution 5x5 |	1x1 stride,  outputs 10x10x16             				|
| RELU            |            |
| Dropout 2       | Used for trianing ONLY                        |
| Pooling         | Output 5x5x16                                 |
| Flatten         | Output 400                                    |
| Fully connected	| Output 120                           									|
| Fully connected	| Output 84                            									|
| RELU            |    |
| Fully connected	| Output 43                            									|


####3. MODEL TRAINING 
Using the AdamOptimizer available in TensorFlow, I stuck with a generic batch size of 128 and really didn't experiment with larger batches nor did I deviate from the default learning rate.  Prior to adding more training data via the rotations -20, -15, -10, 10, 15, 20, I did experiment with the learning rate and batch size - to no avail.

To check for overfitting, I monitored the loss of both the training and validation sets.  The convergence seemed relatively stable.
![alt text][image10]

####4. SOLUTION
In order to get to a solution with a validation set accuracy of at least 0.93, I added a six-fold amount of data to the classes that had less that 1200 samples. Regardless of the number of samples, a passing accuaracy would not be possible if the data were not normalized between -1 and 1 and also preprocessed using the pipeline. Most of the techniques that I used were well know implementations.  The only change to the LeNet architecture that I made was to add dropouts after the first and second RELU activation functions. 

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 94.7% 
* test set accuracy of 93.2%

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]


Here are the results of the prediction:

| ClassID  |      Image      |     Prediction	   	| 
|:--------:|:---------------:|:------------------:| 
|     1    |      30kpm      |   					30kpm   				| 
|     4    |      70kpm     	|        20kpm							|
|    11    |  RoW Next Inter	|  RoW Next Inter				|
|    25    |    Road Work	  	|     Road Work						|
|    33    | Turn Right Ahead| Turn Right Ahead 		|


Originally, the model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares poorly to the accuracy on the test set of 93%.  I believed that I was cropping the original image too closely to the edges, so I resubmitted the same image at 32x32 with slightly larger borders.

Even after re-cropping the image that was failing (70kpm), my accuracy stayed at 80%.  One possibility may be that there was not enough quality training data to classify the 70kpm versus 20kpm - since they are very similar.  Both the 20kpm and 70kpm had under 1200 original samples submitted.  Additional samples were generated by applying small rotation transforms to these originals, so perhaps that was not sufficient.  More importantly, the 0 only had about 200 samples to begin with and the 4 had just over 1200 (the limit for adding samples).  So both ended up with a low amount of final samples.

### SOFTMAX PREDICTIONS
All of the different new samples except for the 70kpm (ClassID 4) reported a prediction probability of over 96% (of those, all were 99% except for the "Turn Right Ahead" ClassID 33). 
This time, the reported probability was almost 37% that the 70kpm (classID 4) was 20kpm (classID 0). The second highest probability was for 30kpm (34%) and then 70kpm (9%).

![alt text][image4]

| Probability         	|     Prediction	        		     |
|:--------------------:|:-----------------------------:| 
| .997        			      | 30kpm   									|
| .001     		          | 80kpm 										|
 							

![alt text][image5] 

| Probability         	|     Prediction	             	|				
|:--------------------:|:----------------------------:| 
| .375        			      | 20kpm   									|
| .347    		           | 30kpm 										|
| .098				             | 70kpm											|
| .089      			        | Bicycles crossing					 				|
| .033			              | 80kpm          |

![alt text][image6] 

| Probability         	|     Prediction	        			 		|
|:--------------------:|:----------------------------:| 
| .999         			     | Right of Way Next Intersection   						|			
| .001     		          | Beware of Ice/Snow 										|

![alt text][image7]

| Probability         	|     Prediction	        			 		|
|:--------------------:|:----------------------------:| 
| 1.000         			    | Road Work   									|


![alt text][image8]

| Probability         	|     Prediction	        				 	|
|:--------------------:|:----------------------------:| 
| .969         			     | Turn Right Ahead   									|
| .014     		          | Ahead Only 										|
| .003					            | Right of Way at Next Intersection						|				
| .003	      		        | Traffic Signals			 				|
| .002				             | 80kpm |





