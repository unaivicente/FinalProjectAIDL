# FinalProjectAIDL

## Introduction

We started our project with a first aim that we proposed, try to use a transformer architecture directly using waveform raw data from our dataset. However, after numerous attemps we ended up trying many different architectures as we understood there were complexities using a transformer and 1D data that were out of our control and knowledge. Therefore, we are presenting in this present work a comparative study between different DL architectures using raw waveform as well as other methods with the same dataset in an attempt to explore sequential data processing and the different feature extractions with convolutional 1D methods.

Our main goal stay the same from our initial aims:
* Classify speech commands in their respective and correct labels
* Tune our model to be the most efficient possible

The deep neural networks used:
* CNN
* LSTM
* GRU

## Tech used

The machine learning framework used is Pytorch and the machine learning environment, Google Colab. Colab provides a GPU with about 12GB of RAM and 38GB of disk to locate the dataset and the .pth file of the model. We don't use Google Cloud because with Colab resources we have enough.

An example of GPU provided by Google Colab, a TESLA K80 GPU with 12GB:

![info_gpu](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/colab_gpu.png)


## Dataset

Some facts about our dataset, identified as SPEECHCOMMANDS version 2 by Google. The WAV audio files are mono (one channel), 16Khz sample rate 1 second human voice recordings of 35 different words. The dataset comprises a sample of: 

| Total audio files  | Total speakers |
| ------------------ | -------------- |
|       126815       |      3124      |

We split the dataset into train, validation and test set. This split isn't random, the dataset contains txt files where specify the wav audio files of each set.

It would be interesting to know the age and gender of the speakers, but for privacy reasons they don't indicate it.

| Split of the dataset | Number of audio files | Number of speakers |
| --- | --- | --- |
| Train           |  105829   |  2618  |
| Validation      |    9981   |   256  |
| Test            |   11005   |   250  |

To downsample the signal from 16kHz to 8kHz we use the torchaudio library.
```
new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=new_sample_rate)
```

To have a visual vision of the audio signal, we plot the waveform of different wav files:

![waveforms](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/waveforms_speech_commands.png)

The deep neural networks can obtain as input a mel spectogram from the audio instead of the raw audio data. A spectogram is a visualization of the frequency spectrum of a signal and the Mel scale mimics how the human ear works (humans are better at detecting differences at lower frequencies than at higher frequencies). We use torchaudio library to get the mel spectogram:

![mel_spectogram](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/zero_mel_spectogram.png)


In our [Dataset_analysis.ipynb](https://github.com/unaivicente/FinalProjectAIDL/blob/main/Dataset_analysis.ipynb) file it is possible to have a deeper understanding of this dataset as well as it is possible to listen to sample examples.

For further information about the dataset, see this [paper](https://arxiv.org/pdf/1804.03209.pdf).

## Experiment

In this experiment, we created different architectures and tuned the hyperparameters of the different deep neural networks in order to find the highest accuracy. We wanted to obtain the different results of the same architectures by changing the sample rate (8kHz or 16kHz) of the input raw data.

We created 2 CNN architectures, 5 LSTM and 6 GRU.

When the models are training, the validation set is used to obtain the val_loss and compare it to the previous val_loss and continue training the model with the lowest loss.

### Different architectures and hyperparameter tuning

#### CNN 

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


#### LSTM

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


#### GRU

Another deep neural network tested is GRU with the following base architecture.

![gru_architecture](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/gru_diagram.png)

Here we present the different GRU architectures tested:

* GRU-1 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 128 hidden dimension units that end up to a fully connected linear layer. Negative log likelihood as loss function adding a LogSoftmax layer in the last layer of the network. Trained with 100 of batch size, 10 epochs, 0.5 of learning rate and SGD optimizer without weight decay.

      - 8kHz accuracy : 90%
      - 16kHz accuracy : 92%

* GRU-2 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 128 hidden dimension units that end up to a fully connected linear layer. Negative log likelihood as loss function adding a LogSoftmax layer in the last layer of the network. Trained with 50 of batch size, 40 epochs, 0.01 of learning rate and SGD optimizer with 0.0001 weight decay and StepLR that decays de learning rate of each parameter group by 0.1 (gamma) every 20 epochs.

      - 8kHz accuracy : 90%
      - 16kHz accuracy : 91%

* GRU-3 : 3 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 256 hidden dimension units that end up to a fully connected linear layer. Cross Entropy Loss as loss function passing the output of the last layer (linear) directly. Trained with 100 of batch size, 10 epochs, 0.5 of learning rate and SGD optimizer without weight decay.
      
      - 8kHz accuracy : 92%
      - 16kHz accuracy : 90%

* GRU-4 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 256 hidden dimension units that end up to a fully connected linear layer. Cross Entropy Loss as loss function passing the output of the last layer (linear) directly. Trained with 100 of batch size, 10 epochs, 1e-4 of learning rate and Adam optimizer.

      - 8kHz accuracy : 89%
      - 16kHz accuracy : 91%
     
* GRU-5 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 128 hidden dimension units that end up to a fully connected linear layer. Negative log likelihood as loss function adding a LogSoftmax layer in the last layer of the network. Trained with 150 of batch size, 10 epochs, 0.01 of learning rate and SGD optimizer without weight decay.

      - 8kHz accuracy : 73%
      - 16kHz accuracy : 64%
 
* GRU-6 : 2 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 256 hidden dimension units that end up to a fully connected linear layer. Cross Entropy Loss as loss function passing the output of the last layer (linear) directly. Trained with 100 of batch size, 30 epochs, 1e-4 of learning rate and Adam optimizer.

      - 8kHz accuracy : 85%
      - 16kHz accuracy : 85%

![grugraph](https://user-images.githubusercontent.com/92716609/164751352-881c90e6-e827-403c-9d3c-07bea17950a6.png)


## Extra experiment
(MELSpec)

