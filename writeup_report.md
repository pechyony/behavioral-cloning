# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_lane.jpg "center lane driving"
[image2]: ./images/left_lane.jpg "left lane driving"
[image3]: ./images/right_lane.jpg "right lane driving"
[image4]: ./images/orig_image.jpg "orig image"
[image5]: ./images/flipped_image.jpg "flipped image"
[image6]: ./images/orig_image2.jpg "orig image"
[image7]: ./images/rotate1.jpg "rotated image1"
[image8]: ./images/rotate1_explained.jpg "rotated image2"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

    python drive.py model.h5

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 142-146)
The model includes ReLU layers to introduce nonlinearity (code lines 142-146), and the data is cropped (code line 140) and normalized in the model using a Keras lambda layer (code line 141). The convolutional and ReLU layers are followed by 5 dense layers. The topmost layer has one outputs, which is a predicted steering angle.

#### 2. Attempts to reduce overfitting in the model

I used L2 regularization (code lines 142-146 and 148-152) and early stopping (code lines 160-169) in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 167-169). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model and training process have a number of hyperparameters, listed in the table below. This table also lists the best values of hyperparameters, found after many rounds of manual tuning:

| Hyperparameter | Tuned Value        		|     Description	        					|
|:-:|:---------------------:|:-----------------------------:|
| learning_rate | 0.001      		| learning rate of optimization algorithm   							|
| alpha | 0.001 | regularization coefficient |
| n_augmentations | 6 | number of rotated images created from each input image |
| patience | 3 |  maximum number of epochs with no improvement in validation error |
| delta | 0.00001 | minimal change of validation error to be considered as improvement |
| steering_correction | 0.2 | correction of steering angle for side images |

These hyperparameters are defined in lines 25-32 of model.py.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
For details about how I created the training data, see the next section.  

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use convolutional layers, followed by dense layers. The convolutional and ReLU layers supposed to learn to extract features from the image, whereas dense layers supposed to learn to predict the right steering angle.

My first step was to augment sample data by using horizontal flipping, and rotations. In order to prevent overfitting, I split my image and steering angle data into a training and validation set. I stopped training models when I saw that there is no significant improvement of mean squared error over the validation set.

Initially I tried to train LeNet-5 network using multiple datasets:

A) augmented sample data  
B) augmented recorded data  
C) augmented sample data + augmented recorded data   

I tried to train the network both with and without L2 regularization. In all cases the car either didn't pass the bridge or didn't pass the sharp left turn right after the bridge or the sharp right turn after that. At this point I concluded that LeNet-5 is not flexible enough to generate steering model for this track.

I switched to NVIDIA network. Since NVIDIA network has twice more layers than LeNet-5, it is more flexible can learn more patterns than LeNet-5. Previously NVIDIA network was also proven to be able to generate good steering models. I trained NVIDIA network datasets A and B, with and without L2 regularization. In all cases the resulting models has the same problems as the models created by LeNet-5. This might be an evidence that dataset A alone or dataset B alone are not sufficient to train a large NVIDIA network. Then I trained NVIDIA network with a larger dataset C and obtained a model that drove two laps but stuck at the bridge at the third lap. Also the driving was very wobbly. This suggested that L2 regularization of network weights might smooth the driving. Indeed, NVIDIA network trained on dataset C with L2 regularization of weights generated a model with very smooth driving without leaving the road.

#### 2. Final Model Architecture

I took the final model architecture (model.py lines 137-152) from [NVIDIA paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). This network has 2 image  preprocessing layers, 5 convolutional layers, 5 ReLU layers and 5 dense layers. The full specification of the network in shown in the table below.

| Layer Number | Layer         		|     Description	        					|
|:-:|:---------------------:|:-----------------------------:|
| 0 | Input         		| 160x320x3 RGB image   							|
| 1 | Cropping   | removes top 70 and bottom 25 rows, outputs array 65x320x3                   |
| 2 | Normalization  | x/255-0.5 transformation, outputs array 65x320x3   |
| 3 | Convolution 5x5   | 24 convolutions, 2x2 stride, valid padding, outputs array 31x158x24 	|
| 4 | ReLU					| 31x158x24 ReLU activation units 												|
| 5 | Convolution 5x5	      	| 36 convolutions, 2x2 stride,  valid padding, outputs array 14x77x36 				|
| 6 | ReLU	    |  14x77x36 ReLU activation units 	 |
| 7 | Convolution 5x5        | 48 convolutions, 2x2 stride,  valid padding, outputs array 5x37x48   |
| 8 | ReLU	      	| 5x37x48 ReLU activation units  |
| 9 | Convolution 3x3        | 64 convolutions, 1x1 stride,  valid padding, outputs array 3x35x64   |
| 10 | ReLU	      	| 3x35x64 ReLU activation units  |
| 11 | Convolution 3x3        | 64 convolutions, 1x1 stride,  valid padding, outputs array 1x33x64   |
| 12 | ReLU	      	| 1x33x64 ReLU activation units  |
| 13 | Flattening          | outputs a vector of length 2112 |
| 14 | Fully connected		| outputs a vector of length 1164        									|
| 15 | Fully connected  | outputs a vector of length 100 |
| 16 | Fully connected   | outputs a vector of length 50                         |
| 17 | Fully connected  | outputs a vector of length 10  |
| 18 | Fully connected   | outputs a scalar |

In Keras, convolution layer includes a nonlinear activation function. Hence the layers 3 and 4 correspond to a single layer in Keras. Similarly each of the pairs of layers 5 and 6, 7 and 8, 9 and 10, 11 and 12 corresponds to a single layer in Keras.

#### 3. Creation of the Training Set & Training Process

I started with the sample dataset provided by Udacity. To capture good driving behavior, I also recorded four laps on track one using center lane driving. Here is an example image of center lane driving, taken from center, left and right cameras:

![alt text][image1]
![alt text][image2]
![alt text][image3]

After combining these two datasets, I obtained 73149 images. I added 0.2 radians to the steering angle of images from the left camera and subtracted 0.2 radians from the steering angle of images from the right camera. This heuristic compensated for the fast that these images were not taken from the center of the car.

The combined dataset is biased towards driving counterclockwise. To remove this bias, I flipped images vertically. For example, here is an original image and the one that has then been flipped:

![alt text][image4]
![alt text][image5]

After the flipping, I obtained a dataset of 73149\*2=146298 images. I split this dataset into training and validation sets, using the ration 80%:20%. In this split all original and flipped images from all cameras that were taken at the same time are either in training and validation set. After the split I obtained  146298*0.8=117038 training images and 146298-117038=29260 validation images.

The training images are focused on the normal driving, where the car drives towards the center of the road. Unfortunately these images are not sufficient for training a robust driving model. Many times the car starts to drive towards the side of the road. To train the model to recover from such state and steer the car toward the center, I augmented the dataset with rotated images. Specifically, for each of 117038 images I created additional 6 rotated images. The rotation angles were chosen uniformly from the range of [-45,45] degrees. Here is an example of the original images and the rotated one:

![alt text][image6]
![alt text][image7]

In the original image the black sides of the road are at the same distance from the center of the image. However in the rotated image the left black side is closer to the center of the image than the right one.
This means that the car is closer to the left side than to the right one. This might be a dangerous state. To return to the center of the road of the road the car needs to increase its steering angle.

We used getRotationMatrix2D function of OpenCV to rotate images. This function rotates images counterclockwise. However the steering angles are measures clockwise. To steer the wheel towards the center of the road, the steering angle of the rotated image is the steering angle used in getRotatationMatrix2D + steering angle in the image before rotation.

After creating 6 rotated images for each of the 117038 images, I had a training set with 117038*7=819266 images. Notice that I did not create rotated images in the validation set. This didn't hurt the driving performance and reduced the training time.

I then preprocessed training and validation sets by removing the top 70 rows of the image, which usually are landscape and are not part of the road. I also removed the bottom 25 rows of the image, which usually are the hood of the car and the road without sidelines. In both cases these rows are not helpful to decide which angle to steer. Finally, I also normalized images by dividing pixel values by 255 and subtracting 0.5. After this normalization the mean value of each pixel is approximately 0. A data with zero mean usually speeds up the training of neural network.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used early stopping mechanism to decide when to stop training. After each epoch I computed mean square error over the validation set and stored the model that generated the lowest validation error so far. If in the last 4 iterations the lowest mean square error did not decrease by at least 0.00001 then the training was stopped. The final model was the one that generated the lowest validation error. I used an adam optimizer with the learning rate 0.001. To prevent overfitting and make the drive more smooth, I used L2 regularization for all weights and biases of the network. I used the same regularization coefficient=0.001 for all weights and biases. Both learning rate and regularization coefficient were tuned to optimize driving over the first track.

Due to the early stopping mechanism, the training was stopped after eleven epochs and the final model was the one created after the seventh epoch. That model has training mean square error of 0.0420 radians and validation mean square error of 0.0432 radians.

## Conclusion

* Rotation of images is an effective technique for training neural network to recover from dangerous situations
* NVIDIA network is powerful enough to generate good steering models
* Since NVIDIA network is very deep, it can be successfully trained only with large datasets. In my case I had to use a training set with 800K examples to train the network
* L2 regularization of weights is very helpful in generating a model that drives smoothly  

## Next Steps
I tried to use the model to drive track 2, but the car crashed after several seconds. This is not surprising since the model was not trained over that track. I plan to create training data for track 2 and train model that drives well in both tracks.  
