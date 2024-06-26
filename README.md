# Project Report: Anomaly Detection in Images using Autoencoder

## Introduction
This project aims to develop an intelligent system capable of detecting anomalous images, such as spam or irrelevant pictures, in a given folder. The system leverages an autoencoder neural network to learn the normal image patterns and identify outliers based on reconstruction errors. The project involves loading and preprocessing images, training an autoencoder model using TensorFlow and Keras, and identifying anomalies by setting a threshold for reconstruction errors.

## Image Loading and Preprocessing
### Function
A function was created to load and preprocess images from a specified folder. The key steps include:
- **Reading Images**: Loading images with valid extensions (.jpg, .jpeg, .png, .gif, .bmp) from the folder.
- **Resizing**: Resizing images to a consistent size (128x128) for uniformity.
- **Normalization**: Normalizing pixel values to a range of 0 to 1 for better model performance.

## Data Splitting
The dataset is split into training and testing sets using an 80-20 split ratio to ensure a balanced evaluation. This allows the model to learn from the majority of the data while being tested on a separate subset to evaluate its performance.

## Autoencoder Model
### Architecture
An autoencoder model was designed with an encoder and a decoder:
- **Encoder**: The encoder compresses the input image into a latent space representation. This involves flattening the input, followed by dense layers to reduce the dimensionality.
- **Decoder**: The decoder reconstructs the image from the latent space representation. This involves dense layers to expand the dimensionality back to the original image shape and reshaping to match the input format.

### Latent Dimension
The latent dimension, a key parameter for the autoencoder, was set to 32. This dimension can be adjusted based on the complexity and variability of the input images.

### Compilation
The autoencoder model was compiled using the Adam optimizer and Mean Squared Error (MSE) loss function. This combination is effective for training the model to minimize reconstruction errors.

## Training
The autoencoder was trained on the training set for 20 epochs with a batch size of 32. The training process involved shuffling the data and using a validation split to monitor the model's performance on unseen data.

## Anomaly Detection
### Reconstruction Errors
After training, the autoencoder was used to reconstruct the images in the test set. Reconstruction errors were calculated as the mean squared error between the original and reconstructed images.

### Threshold Setting
A threshold for anomaly detection was set based on the reconstruction errors. The threshold was determined as the mean reconstruction error plus 0.5 times the standard deviation. This helps in distinguishing between normal and anomalous images.

### Anomaly Identification
Images with reconstruction errors exceeding the threshold were identified as anomalies. These images are considered to deviate significantly from the learned normal patterns and are flagged for further review or deletion.

## Conclusion
This project successfully developed an autoencoder-based system for detecting anomalous images in a folder. The combination of image preprocessing, a well-designed autoencoder model, and effective threshold setting enabled the identification of outliers. Future improvements could involve fine-tuning the model, experimenting with different architectures, and incorporating additional preprocessing steps to enhance performance.
