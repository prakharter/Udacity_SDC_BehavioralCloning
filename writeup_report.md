# Udacity-Behavioral-Cloning

 
This project uses convolutional neural network to predict steering angle from image and drive a car in the simulator.
  

# Overview
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

### Project Files
|  Filename   |   Description  | 
|:-------------:|:-------------:|
| preprocess.py |  python file for data preprocessing and argumentation |
| train.py | define and train the neual network |
| model.h5 | saved model by keras |
| drive.py | communicate with simulator and use saved model to predict steering angle  |
| video.ogv | track 1 video record |

### Usage
Download simulator from [thie repository](https://github.com/udacity/self-driving-car-sim), run the simulator in 
autonomous mode and execute following command:
```
> python drive.py model.h5
```

### Data Preprocessing & Argumentation
Here I use the [Udacity sample data] present in the worksapce data folder. 
  
After dropping 80% of data with 0 steering angle, left & right camera images are used with angle correction and data argumentation is
applied to the center image. Images are cropped and resized to 75x320x3 shape. For each row of data in csv file, 8 images are generated (or 7 since image with 0 steering angle won't
be flipped):  
  

Here I tried different distribution by subsampling the generated data and decided to use all generated data. 

### Model Architecture and Training
The model is based on [Nvidia's paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) 
with following modification:
* use input shape of 75x320x3 instead of 66x200x3
* use rgb channel instead of yuv
* remove first fully connected layer with 1164 neurons
* add a dropout layer to avoid overfitting
* use elu instead of relu as activate function of covolution layer

The training uses mean squared error as cost function and Adam optimizer with 0.001 learning rate,
10% data as validation data, 5 epochs and batch size of 32.

  
