# VisionMath
AI-Powered simple mathematical expression solver

## Description
VisionMath takes a handwritten mathematical operation and uses AI to recognize the characters. Afterwards, it builds the operation and returns the corresponding value.

## Installation
Add your image to the images folder and run visionmath.py

## Tests
<p><b>Input image</b></p>
<img src="https://github.com/AdrianTorremochaUMA/VisionMath/blob/main/images/test15.jpeg?raw=true" height=300 width=400>
<p><b>Left-Right ordered contours image</b></p>
<img src="https://github.com/AdrianTorremochaUMA/VisionMath/blob/main/images/testResults/test15_contours.png?raw=true" height=300 width=400>
<p><b>Model classification</b></p>
<img src="https://github.com/AdrianTorremochaUMA/VisionMath/blob/main/images/testResults/test15_classification.png?raw=true" height=150 width=600>

### Image Output
<p>EXPRESSION: ((4-2)/(6)) </p>
<p>RESULT: 0.33 </p>

## Keras Model
<p>VisionMath uses a Tensorflow Keras model formed by:</p>
<p></p>
<p>1 - Flatten Layer (input_shape = (28,28))</p>
<p>2 - Dense Layer (units = 128, activation = relu)</p>
<p>3 - Dense Layer (units = 128, activation = relu)</p>
<p>4 - Dense Layer (units = 13, activation = softmax)</p>
<p></p>
<p>Validation test accuracy: 100%</p>


