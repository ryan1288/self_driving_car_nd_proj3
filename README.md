# **Traffic Sign Recognition Project** 
---
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: report_images/sign_12.png
[image2]: report_images/training_hist.png
[image3]: report_images/valid_hist.png
[image4]: report_images/test_hist.png
[image5]: report_images/signs.png
[image6]: report_images/feature_map.png

---
## Data Set Summary & Exploration

### Basic Data Set Summary

Using the numpy library [Code Block 2], I calculated:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3) - 32x32 image with 3 layers (RGB)
* The number of unique classes/labels in the data set is 43

### Exploratory Visualization
First, a sample image is provided along with the label:

[Code Block 3]

Label: 12 - Priority Road

![][image1]

Then, histograms showing the tributions of the images are below:
The distributions are similar despite the different data set sizes.
![][image2]
![][image3]
![][image4]

## Design and Test a Model Architecture

### Data Preprocessing
Testing with the basic normalization, the model was able to successfully reach validation rates of up to 95%, hence no further preprocessing was done.

Normalizing was done [Code Block 4] to center the data about 0, streamlining the process and increasing the likelihood of a better model while optimizing.

`
X_train = X_train/128 - 1
X_validation = X_validation/128 - 1
X_test = X_test/128 - 1
`

The data of 8 bits was normalized to be between -1 to 1 instead of 0 to 255.

### Model Architecture
My final model [Code Block 5] consisted of the following 13 layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16 					|
| Flattening	        | Array of 400 (= 5x5x16) 						|
| Fully connected		| Output of size 120							|
| RELU					|												|
| Dropout				|												|
| Fully connected		| Output of size 84								|
| RELU					|												|
| Dropout				|												|
| Fully connected		| Output of size 43								|

 
### Training the Model
Hyperparameters:
* Epochs: 15
* Learning rate: 0.0005
* Batch size: 32
* Training dropout rate: 40%

A training pipeline [Code Blocks 6-9] was created to train the model:
* Set up `tf.placeholder` input/output TensorFlow variables with one hot encoding
* Set hyperparameters including learning rate, epochs, batch sizes, and dropout probability
* Set `tf.truncated_normal()` mean and standard deviation used for variable initialization
* Use softmax to set cross entropy
* Calculate the loss
* Use the Adam Optimizer to make weight and bias adjustments
* Establish the overall training operation
* Iterate through epochs and shuffled batches of training features and labels at a 40% dropout rate
* In each epoch, the average accuracy is calculated through the validation set with 0% dropout rate

### Results and Discussion
My final model results [Code Blocks 9-10] were:
* training set accuracy of 0.996
* validation set accuracy of 0.963 
* test set accuracy of 0.943

LeNet was a great starting point for the classification of traffic signs because it yielded great success in identifying numbers, a similar problem. Traffic signs are arguably simpler than handwriting to identify, as traffic signs have a fixed shape despite different environments and shades. The final results indicate that LeNet already provided good results, but further adjustments made it even more accurate in determining the correct traffic sign.

Starting with LeNet, it provided accuracies of ~90-91% accuracy, which was insufficiently accurate. Next, I added the dropout step to both fully connected layers and iterated through different batch sizes, epochs, learning rates, and dropout rates. 

As batch sizes went down, training time and accuracy went up. As epochs increased, the accuracy also went up until it saturates 9evidence by the training accuracy reaching near 100%). learning rates' effect varied, but in general, if the testing data's accuracy does not saturate, it continues to improve the resultant accuracy. Dropout rate has dimishing returns if set too high, and 40% was found to be a good middleground.

A dropout layer is useful because it the model's reliance on specific weights in each layer by completely turning them off, this forces the model to gradaully be more generalized, leading to lower overfitting and better results. A convolution layer drastically reduces the amount of time required to train the model due to the lessened number of parameters compared to a fully connected layer. Also, it helps minimize the effects of the traffic sign's position and orientation in the picture as the weights are shared on each layer.

While the training set accuracy is greater than the validation set accuracy, the overfitting is not severe as the difference is only 3%.

## Test a Model on New Images

### Traffic Signs from the Web
Traffic Sign Model Prediction [Code Block 13]

Here are five German traffic signs that I found on the web:

![][image5]

These images may be difficult to classify (left to right) because:
1. Includes two semicircles on top, which may mistlead the model into thinking that those are numbers.
2. Has white strips in the background that match in color with the arrow.
3. The bottom is intentionally cut off while reformatting the image, and the model may not be familiar with incomplete signs.
4. A good baseline test with primarily the sign.
4. Includes a different shape in the background (different sign) and also the road name sign on top.

### Model's Predictions and Comparisons
Here are the results of the prediction [Code Block 14-16] (left to right):

| Image			        |     Prediction		| 
|:---------------------:|:---------------------:| 
| Yield					| Yield					| 
| Turn Left Ahead		| Turn Left Ahead		|
| Roundabout Mandatory	| Roundabout Mandatory	|
| Go Straight or Right	| Go Straight or Right	|
| Priority Road			| Priority Road			|

The model was able to correctly guess all 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy of the test sets. However, on a different model testing run, the model only guessed 3 of the 5 accurately, so the consistency can be improved by adjusting the architecture and hyperparameters when noticed.

### Model Prediction Confidence using Softmax Propabilities
Prediction Code Cell [Code Block 16]

Below, the top 5 Softmax probabilities are shown as well as their labels:

First image: the model is confident that this is a yield sign (probability of 1), and the image does contain a yield sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield Sign   									| 

Second image: the model was fairly certain that it was a turn left ahead sign, but it was also consider the keep right option (potentially due to the arrows).

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.76         			| Turn Left Ahead   							| 
| 0.22     				| Keep Right 									|
| 0.01					| Go Straight or Go Right						|
| <0.01	      			| Turn Right Ahead								|
| <0.01				    | Ahead Only      								|

Third image: the model was a somewhat sure it was a roundabout image, but the rotating arrows likely made it consider the other turning options.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.62         			| Roundabout Mandatory   						| 
| 0.12     				| Keep Right 									|
| 0.12					| Turn Left Ahead								|
| 0.10	      			| Turn Right Ahead					 			|
| 0.02				    | Go Straight or Go Right      					|

Fourth image: the model was certain that it is a go straight or go right sign, with other possibilities under 10e-7.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Go Straight or Go Right   					| 

Fifth image: the model was certain that it is a priority road sign, with other possibilities under 10e-24.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority Road   								| 


## Visualizing the Neural Network
Visualizing the model [Code Block 17] after the second pooling and convolution layer, the images below show that the model differentiates the yield sign's corners, triangular shape, and edges. Some images are black, potentially screening out other possibilities.

![][image6]

