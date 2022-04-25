# MELSPECTOGRAM

## EXTRA EXPERIMENT PLAYING WITH MELSPECTOGRAM TRANSFORMATION

We were encouraged to try another kind of model using MelSpectogram transformation, witch was included in torchaudio library. The idea was to adapt the template that we used in CNN, GRU and LSTM just changing the model architecture and the data input, because this time we are working with images. We used the model architecture from this [repository](https://github.com/aminul-huq/Speech-Classification) which is based on this [web](https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/) using a different dataset.

Here we present the following model architectures based only on MelSpectogram, this time the input will be 2-dimensional:

![m1_diagram](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/M1_diagram.PNG)

- M1: Using 8Khz data.
We use 2 Convolutional 2D Neural Network to extract the abstract features because we are working with images. In the middle of the Conv2d we'll use two dropouts (0.5). The main purpose of it is to avoid the overfitting of the model as long as possible. Finally we use 2 Linear Layers and a third one as the output with the total number of classes (35).
Accuracy: 64%

- M1: Using 16Khz data.
The model architecture is the same as in 16kHz. Here we are interested in how the downsampling has affected to the accuracy of the model. As we can see, the downsampling has improved the model for about an 10%.
Accuracy: 55%

![m1_train_loss](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/training_loss_M1.PNG)

![m2_diagram](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/M2_diagram.PNG)

- M2: Using 8Khz data.
This model has the purpose to see the changes over the hyperparameters. In this case we have changed the number of batches (from 256 to 100) and the learning rate (from 0.01 to 0.05)
Accuracy: 62%

- M2: Using 16Khz data.
In this case, although we have conserved the same hyperparameters from the downsampled model M2, there is not (practically) any improvement of the model.
Accuracy: 63%

![m2_train_loss](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/training_loss_M2.PNG)

![m3_diagram](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/M3_diagram.PNG)

- M3: Using 8Khz data.
The idea of the last model was to change a bit the architecture of the model in the downsampled version. In this case we decided to delete one of the dropouts of 0.5 and replacing it with another one of 0.3 but after the first Linear Layer (1664,256) or (128+12x32x3,256).
For the moment is the best accuracy that has been obtained.
Accuracy: 71%

- M3: Using 16Khz data.
In this case, we work with all the samples and we put all the dropouts of 0.3 between all the layers but the accuracy doesn't seem to improve.
Accuracy: 60%

![m3_train_loss](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/training_loss_M3.PNG)

As an extra note, adding the number of epochs wasn't improve the accuracies either.

## REFERENCES

- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf)
- [Building an End-to-End Speech Recognition Model in PyTorch](https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/)

