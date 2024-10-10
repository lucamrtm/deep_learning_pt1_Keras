# Fashion MNIST Classification with TensorFlow and Keras

This project implements a neural network using TensorFlow and Keras to classify images from the Fashion MNIST dataset. The Fashion MNIST dataset contains grayscale images of 10 different types of clothing items, such as T-shirts, trousers, and shoes. The goal of this project is to train a model that can accurately classify these items of clothing.

## Project Overview

The code provided demonstrates the following:

- **Data loading and exploration**: Loading the Fashion MNIST dataset, inspecting the training and testing sets, and visualizing sample images.
- **Data normalization**: Normalizing the pixel values of the images to improve model performance.
- **Model creation**: Building a sequential neural network model using Keras. The architecture includes:
  - A flattening layer to convert 28x28 pixel images into a 1D array.
  - A dense layer with 256 neurons and ReLU activation.
  - A dropout layer to reduce overfitting.
  - A final dense layer with 10 neurons and softmax activation for classification.
- **Model training**: The model is compiled with the Adam optimizer and trained using sparse categorical cross-entropy as the loss function. Accuracy metrics are tracked over 5 epochs.
- **Model evaluation**: Evaluating the trained model on the test set and comparing performance through accuracy and loss graphs.
- **Saving and loading the model**: The trained model is saved to a file for future use and reloaded for testing.

## Dataset

The [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) is used for training and evaluation. It consists of 70,000 images (60,000 for training and 10,000 for testing) of fashion items, each labeled as one of 10 categories:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Key Concepts

1. **Data Normalization**: The images are scaled from the original range of [0, 255] to [0, 1] to speed up training and improve the model's convergence.
 
2. **Sequential Model**: This simple feed-forward architecture is built using Keras' Sequential API, making it easy to stack layers.

3. **Dropout Regularization**: A dropout rate of 20% is applied to prevent overfitting by randomly dropping neurons during training.

4. **Model Compilation**: The model uses the Adam optimizer and is compiled with sparse categorical cross-entropy loss. 

5. **Training and Validation**: The model is trained for 5 epochs with a validation split of 20% to monitor performance on unseen data during training.

## Libraries Used

- **TensorFlow**: The main deep learning framework used to build and train the model.
- **Keras**: A high-level API that simplifies the creation of neural networks.
- **Matplotlib**: Used to visualize accuracy and loss over the training epochs.
- **NumPy**: For numerical operations on the dataset.

## Results

The model's performance is visualized through accuracy and loss graphs:

- **Accuracy per Epoch**: Shows how the model's accuracy improves over time.
- **Loss per Epoch**: Tracks the reduction in training and validation loss, showing how the model learns.

## How to Run

### Option 1: Use Google Colab

1. Open the Colab notebook by clicking the following link: [Google Colab Notebook](https://colab.research.google.com/drive/1I8ppav28RYixm64hfE0gQ-F_Vuz1sBTe?usp=sharing)
2. Once the notebook is open, you can run each cell in the notebook sequentially.
3. All dependencies will be automatically installed within Colab, and the dataset will be loaded and trained on the platform.

### Option 2: Run Locally

1. Clone this repository:
    ```bash
    git clone https://github.com/lucamrtm/deep_learning_pt1_Keras.git
    cd fashion-mnist-classification
    ```

2. Install dependencies:
    ```bash
    pip install tensorflow matplotlib numpy
    ```

3. Run the script:
    ```bash
    python fashion_mnist_classification.py
    ```

4. After training, the model will be saved to a file called `modelo_epochs5_nos3.h5`. You can load and test the model using the saved file.



## Future Improvements

- Tune parameters such as learning rate, number of neurons, and epochs.
- Experiment with more advanced neural network architectures, such as convolutional neural networks (CNNs), to improve classification performance.
- Incorporate more sophisticated data techniques to further reduce overfitting.





