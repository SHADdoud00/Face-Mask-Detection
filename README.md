# Face-Mask-Detection
Detecting face masks using Python, Keras, OpenCV on real video streams
# Introduction

This tutorial will show you how to use Python, Keras, and OpenCV to detect face masks in real video streams. We will be using a pre-trained CNN model to detect the masks in the video stream. The model was trained using a dataset of over 2000 images of people with and without face masks.

# Prerequisites

* Python (3.6 or higher)
* Keras 
* OpenCV

# Setup

1. Install the necessary packages (Python, Keras, and OpenCV).
2. Download the pre-trained CNN model from [link].
3. Unzip the model into a folder.

# Code

1. Import the necessary libraries:

```python
import cv2
import numpy as np
from keras.models import load_model
```

2. Load the pre-trained model:

```python
model = load_model('model.h5')
```

3. Initialize the video capture:

```python
cap = cv2.VideoCapture(0)
```

4. Read each frame from the video stream and predict
