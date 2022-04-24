
Here we present the following model architectures based only on MelSpectogram, this time the input will be 2-dimensional:
- M1: Using 8Khz data.
We use 2 Convolutional 2D Neural Network to extract the abstract features because we are working with images. In the middle of the Conv2d we'll use two dropouts (0.5). The main purpose of it is to avoid the overfitting of the model as long as possible. Finally we use 2 Linear Layers and a third one as the output with the total number of classes (35).
Accuracy: 64%

- M1: Using 16Khz data.
The model architecture is the same as in 16kHz. Here we are interested in how the downsampling has affected to the accuracy of the model. As we can see, the downsampling has improved the model for about an 10%.
Accuracy: 55%

- M2: Using 8Khz data.
This model has the purpose to see the changes over the hyperparameters. In this case we have changed the number of batches (from 256 to 100) and the learning rate (from 0.01 to 0.05)
Accuracy: 62%

- M2: Using 16Khz data.
In this case, although we have conserved the same hyperparameters from the downsampled model M2, there is not (practically) any improvement of the model.
Accuracy: 63%

- M3: Using 8Khz data.
The idea of the last model was to change a bit the architecture of the model in the downsampled version. In this case we decided to delete one of the dropouts of 0.5 and replacing it with another one of 0.3 but after the first Linear Layer (1664,256) or (128+12x32x3,256).
For the moment is the best accuracy that has been obtained.
Accuracy: 71%

- M3: Using 16Khz data.
In this case, we work with all the samples and we put all the dropouts of 0.3 between all the layers but the accuracy doesn't seem to improve.
Accuracy: 60%

As an extra note, adding the number of epochs wasn't improve the accuracies either. 
