# FinalProjectAIDL

# INTRODUCTION

We started our project with a first aim that we proposed, try to use a transformer architecture directly using waveform raw data from our dataset. However, after numerous attemps we ended up trying many different architectures as we understood there were complexities using a transformer and 1D data that were out of our control and knowledge. Therefore, we are presenting in this present work a comparative study between different DL architectures using raw waveform as well as other methods with the same dataset in an attempt to explore sequential data processing and the different feature extractions with convolutional 1D methods.
Our main goal stay the same from our initial aims:
* Classify speech commands in their respective and correct labels
* Tune our model to be the most efficient possible

# METHODS

Some facts about our dataset, identified as SPEECHCOMMANDS version 2 by Google, 16Khz sample rate 1 second human voice recordings of 35 different words. The dataset comprises a sample of: 
  Total audio files: 126815
  
  Audio files in training: 105829 
  
  Audio files in validation: 9981 
  
  Audio files in test: 11005 
  
  Total speakers: 3124 
  
  Speakers in training: 2618 
  
  Speakers in validation: 256 
  
  Speakers in testing: 250

In our Results.ipynb(FALTA LINK) file it is possible to have a deeper understanding of this dataset as well as it is possible to listen to sample examples.

# CNN ARCHITECTURES

Here we present the following model architectures based only on 1D Convolutional Layers:

C1: Using 8Khz data. 4 layer 1D convolutional layers (Conv1d) that go from 1 to 64 channels, with batch normalization and 1D maxpooling (Maxpool1d). It is the reference and it is directly extracted from M5 model in https://arxiv.org/pdf/1610.00087.pdf

Accuracy: 77%
C1: Using 16Khz data. 4 layer 1D convolutional layers (Conv1d) that go from 1 to 64 channels, with batch normalization and 1D maxpooling (Maxpool1d). We use the exact same M5 architecture as before and we decide to stay with 16Khz data as it seems like can have a better feature extraction.

Accuracy: 83%
C2: Using 8Khz data. 6 layer 1D convolutional layers with extended channels from 1 to 2048, also same batch normalization and 1D maxpooling until layer 4.

Accuracy: 80%
C2: Using 16Khz data. 6 layer 1D convolutional layers with extended channels from 1 to 2048, also same batch normalization and 1D maxpooling until layer 4.

Accuracy: 85%

![cnngraph](https://user-images.githubusercontent.com/92716609/164751341-ef5ebeb8-6da2-4533-b50d-871e4ec74019.png)


##1 LSTM ARCHITECTURES

Here we present the following model architectures:

L1: Using 16Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling and then connecting it to a LSTM with 128 hidden dimension units that end up to a fully connected linear layer.

Accuracy: 92%
L2: Using 16Khz data. Same as L1 but with 256 hidden layers in the LSTM layer.

Accuracy: 92%
L3: Using 16Khz data. 3 layer 1D convolutional layers in the same sequence as in L1.

Accuracy: 88%
L4: Using 16Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling, we add another 5th layer with maxpooling and then connect 1024 channels to a LSTM with 128 hidden dimension units that end up to a fully connected linear layer.

Accuracy: 91%
L5: Using 16Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling, we add another two layers (5th and 6th) with maxpooling and then connect 2048 channels to a LSTM with 256 hidden dimension units that end up to a fully connected linear layer.

Accuracy: 91%
L1: Using 8Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling and then connecting it to a LSTM with 128 hidden dimension units that end up to a fully connected linear layer.

Accuracy: 90%
L2: Using 8Khz data. Same as L1 but with 256 hidden layers in the LSTM layer.

Accuracy: 90%
L3: Using 8Khz data. 3 layer 1D convolutional layers in the same sequence as in L1.

Accuracy: 88%
L4: Using 8Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling, we add another 5th layer with maxpooling and then connect 1024 channels to a LSTM with 128 hidden dimension units that end up to a fully connected linear layer.

Accuracy: 79%
L5: Using 8Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling, we add another two layers (5th and 6th) with maxpooling and then connect 2048 channels to a LSTM with 256 hidden dimension units that end up to a fully connected linear layer.

Accuracy: 88%

![lstmgraph](https://user-images.githubusercontent.com/92716609/164751348-10fe9c45-efdb-464f-b99e-46fe77e1f1cc.png)

![grugraph](https://user-images.githubusercontent.com/92716609/164751352-881c90e6-e827-403c-9d3c-07bea17950a6.png)

