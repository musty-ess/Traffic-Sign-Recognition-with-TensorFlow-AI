# Traffic Sign Recognition with TensorFlow

## Overview

This project implements a neural network using TensorFlow to classify images of traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model accurately identifies different types of traffic signs, such as stop signs, speed limit signs, and yield signs, among others.

## Getting Started

To begin, ensure you have Python 3.12 installed, as specified for compatibility with TensorFlow. Follow these steps to set up the project:

1. **Clone the repo**: `git clone https://github.com/musty-ess/Traffic-Sign-Recognition-with-TensorFlow-AI.git`

2. **Download the GTSRB dataset in this directory**: `https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip`

3. **Install Dependencies**:
   Navigate to the `traffic` directory and run: `pip install -r requirements.txt`

   This installs necessary dependencies like `OpenCV-Python` for image processing, `scikit-learn` for machine learning functions, and `TensorFlow` for neural networks.

4. **Run the program**: `python traffic.py gtsrb [name_of_model.h5]`



## Implementation Details

### `traffic.py`

The core of the project resides in `traffic.py`, where we implement two key functions:

- **`load_data(data_dir)`**: This function loads the image data and corresponding labels from the specified `data_dir`. Each image is resized to a standard size (`IMG_WIDTH x IMG_HEIGHT`) using OpenCV-Python (`cv2`) and converted into a numpy array. It returns two lists: `images` containing the image arrays and `labels` containing the integer category for each image.

- **`get_model()`**: This function builds and compiles a neural network model using TensorFlow's Keras API. The model architecture is customizable, allowing experimentation with various configurations of convolutional and pooling layers, hidden layers, and dropout to optimize accuracy.

### Training and Evaluation

Once the data is loaded and the model is built, `traffic.py` trains the model on the training set and evaluates its performance on the testing set. The training progress, including loss and accuracy metrics for each epoch, is displayed.

## Experimentation

Throughout the project, experimentation with different model architectures and hyperparameters is encouraged. You can modify `get_model()` to explore:

- Different numbers of convolutional and pooling layers.
- Varying sizes and numbers of filters in convolutional layers.
- Various configurations of hidden layers and dropout rates.

### Observations

During experimentation, observe the effects of these changes on training time, convergence, and model accuracy. Document what configurations yield the best results and any challenges encountered in achieving optimal performance.

## Conclusion

Building a traffic sign recognition system involves leveraging TensorFlow's capabilities to create and train neural networks. By experimenting with different architectures and configurations, we aim to improve the model's accuracy in identifying traffic signs, contributing to advancements in computer vision for autonomous vehicles.

## Acknowledgements
Data provided by J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011
