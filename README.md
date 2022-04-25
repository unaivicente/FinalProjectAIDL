# Speech Classification
This is a project with several Deep Learning models that receive an audio input and predict the corresponding label. 

We started our project with a first aim that we proposed, try to use a transformer architecture directly using waveform raw data from our dataset. However, after numerous attemps we ended up trying many different architectures as we understood there were complexities using a transformer and 1D data that were out of our control and knowledge. Therefore, we are presenting in this present work a comparative study between different DL architectures using raw waveform as well as other methods with the same dataset in an attempt to explore sequential data processing and the different feature extractions with convolutional 1D methods.

Our main goal stay the same from our initial aims:
* Classify speech commands in their respective and correct labels
* Tune our model to be the most efficient possible

The deep neural networks tested:
* CNN
* LSTM
* GRU

We found a [Kaggle challenge](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/leaderboard?) on speech recognition with the same dataset as ours and this encouraged us to try to achieve a higher accuracy than the winner.

![leaderboard_kaggle](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/leaderboard_kaggle.png)



## Tech used

The machine learning framework used is Pytorch and the machine learning environment, Google Colab. Colab provides a GPU with about 12GB of RAM and 38GB of disk to locate the dataset and the .pth file of the model. We don't use Google Cloud because with Colab resources we have enough.

An example of GPU provided by Google Colab, a TESLA K80 GPU with 12GB:

![info_gpu](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/colab_gpu.png)


## Usage

### Folder structure
The first 4 folders contains the different models, the 'doc' folder contains the images, graphs and visual documentation, the 'Dataset_analysis.ipynb' file shows dataset features and 'Results.ipynb' upload the .pth models and prepare it to pass the test set of the dataset and view the results obtained.
```
├── CNN
├── GRU
├── LSTM
├── MEL
├── doc
├── Dataset_analysis.ipynb
├── Results.ipynb
├── README.md

```

CNN folder. Contains 4 .ipynb files of models to train them and a README file for more information about the models architecture and hyperparameter tuning.
```
CNN
├── C1_CNN_16kHz.ipynb
├── C1_CNN_8kHz.ipynb
├── C2_6layers_CNN_16kHz.ipynb
├── C2_6layers_CNN_8kHz.ipynb
├── README_CNN.md

```

GRU folder. Contains 12 .ipynb files of models to train them and a README file for more information about the models architecture and hyperparameter tuning.
```
GRU
├── GRU-1_16kHz.ipynb 
├── GRU-1_8kHz.ipynb 
├── GRU-2_16kHz.ipynb 
├── GRU-2_8kHz.ipynb 
├── GRU-3_16kHz.ipynb 
├── GRU-3_8kHz.ipynb 
├── GRU-4_16kHz.ipynb 
├── GRU-4_8kHz.ipynb 
├── GRU-5_16kHz.ipynb 
├── GRU-5_8kHz.ipynb 
├── GRU-6_16kHz.ipynb 
├── GRU-6_8kHz.ipynb 
├── README_GRU.md

```

LSTM folder. Contains 10 .ipynb files of models to train them and a README file for more information about the models architecture and hyperparameter tuning.
```
LSTM
├── L1_4layersREF_LSTM_16kHz.ipynb 
├── L1_4layersREF_LSTM_8kHz.ipynb 
├── L2_4layers256h_LSTM_16kHz.ipynb 
├── L2_4layers256h_LSTM_8kHz.ipynb 
├── L3_3layers_LSTM_16kHz.ipynb
├── L3_3layers_LSTM_8kHz.ipynb
├── L4_5layers_LSTM_16kHz.ipynb 
├── L4_5layers_LSTM_8kHz.ipynb 
├── L5_6layers_LSTM_16kHz.ipynb
├── L5_6layers_LSTM_8kHz.ipynb
├──README_LSTM.md

```

MEL folder. Contains 6 .ipynb files of models to train them and a README file for more information about the models architecture and hyperparameter tuning.
```
MEL
├── M1_MEL_16kHz.ipynb
├── M1_MEL_8kHz.ipynb 
├── M2_MEL_16kHz.ipynb
├── M2_MEL_8kHz.ipynb
├── M3_MEL_16kHz.ipynb
├── M3_MEL_8kHz.ipynb
├──README_MEL.md

```

### To run the files
The only requirement is to have a Google account and download the .ipynb files of this repository.

      1. Download the .ipynb files.
      2. Open Google Colaboratory, click File -> Upload notebook and select all the .ipynb files.
      3. Open the different .ipynb you want to run and the only you have to do is click on Runtime -> Run all.


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

### Hypothesis

### Experiment setup

In this experiment, we created different architectures and tuned the hyperparameters of the different deep neural networks in order to find the highest accuracy. We wanted to obtain the different results of the same architectures by changing the sample rate (8kHz or 16kHz) of the input raw data.

We created 2 CNN architectures, 5 LSTM and 6 GRU.

When the models are training, the validation set is used to obtain the val_loss and compare it to the previous val_loss and continue training the model with the lowest loss.

#### Different architectures and hyperparameter tuning

##### CNN 

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


##### LSTM

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


##### GRU

Another deep neural network tested is GRU with the following base architecture.

![gru_architecture](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/gru_diagram.png)

Here we present the different GRU architectures tested:

* GRU-1 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 128 hidden dimension units that end up to a fully connected linear layer. Negative log likelihood as loss function adding a LogSoftmax layer in the last layer of the network. Trained with 100 of batch size, 10 epochs, 0.5 of learning rate and SGD optimizer without weight decay.

* GRU-2 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 128 hidden dimension units that end up to a fully connected linear layer. Negative log likelihood as loss function adding a LogSoftmax layer in the last layer of the network. Trained with 50 of batch size, 40 epochs, 0.01 of learning rate and SGD optimizer with 0.0001 weight decay and StepLR that decays de learning rate of each parameter group by 0.1 (gamma) every 20 epochs.

* GRU-3 : 3 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 256 hidden dimension units that end up to a fully connected linear layer. Cross Entropy Loss as loss function passing the output of the last layer (linear) directly. Trained with 100 of batch size, 10 epochs, 0.5 of learning rate and SGD optimizer without weight decay.

* GRU-4 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 256 hidden dimension units that end up to a fully connected linear layer. Cross Entropy Loss as loss function passing the output of the last layer (linear) directly. Trained with 100 of batch size, 10 epochs, 1e-4 of learning rate and Adam optimizer.

* GRU-5 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 128 hidden dimension units that end up to a fully connected linear layer. Negative log likelihood as loss function adding a LogSoftmax layer in the last layer of the network. Trained with 150 of batch size, 10 epochs, 0.01 of learning rate and SGD optimizer without weight decay.

* GRU-6 : 2 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 256 hidden dimension units that end up to a fully connected linear layer. Cross Entropy Loss as loss function passing the output of the last layer (linear) directly. Trained with 100 of batch size, 30 epochs, 1e-4 of learning rate and Adam optimizer.


For more information about the different architectures, see [README](https://github.com/unaivicente/FinalProjectAIDL/blob/main/GRU/README_GRU.md) in GRU folder.

### Results
The main objective was to improve the accuracies found in the [M5 paper](https://arxiv.org/pdf/1610.00087.pdf) (63.42% accuracy of M5 model) and in the [Kaggle challenge](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/overview) (91.06% acc.). The accuracies obtained passing the test set are the following:

| Model | 8kHz audio input | 16kHz audio input |
| --- | --- | --- |
| C1    |  77%  |  83%  |
| C2    |  80%  |  85%  |
| L1    |  90%  |  92%  |
| L2    |  90%  |  92%  |
| L3    |  88%  |  88%  |
| L4    |  79%  |  91%  |
| L5    |  88%  |  91%  |
| GRU1  |  90%  |  92%  |
| GRU2  |  90%  |  91%  |
| GRU3  |  92%  |  90%  |
| GRU4  |  89%  |  91%  |
| GRU5  |  73%  |  64%  |
| GRU6  |  85%  |  85%  |

![grugraph](https://user-images.githubusercontent.com/92716609/164751352-881c90e6-e827-403c-9d3c-07bea17950a6.png)

### Conclusions


## Extra experiment
##### MELSPECTOGRAM

Here we present the following model architectures based only on MelSpectogram, this time the input will be 2-dimensional:

M1: Using 8Khz data. We use 2 Convolutional 2D Neural Network to extract the abstract features because we are working with images. In the middle of the Conv2d we'll use two dropouts (0.5). The main purpose of it is to avoid the overfitting of the model as long as possible. Finally we use 2 Linear Layers and a third one as the output with the total number of classes (35). Accuracy: 64%

M1: Using 16Khz data. The model architecture is the same as in 16kHz. Here we are interested in how the downsampling has affected to the accuracy of the model. As we can see, the downsampling has improved the model for about an 10%. Accuracy: 55%

M2: Using 8Khz data. This model has the purpose to see the changes over the hyperparameters. In this case we have changed the number of batches (from 256 to 100) and the learning rate (from 0.01 to 0.05) Accuracy: 62%

M2: Using 16Khz data. In this case, although we have conserved the same hyperparameters from the downsampled model M2, there is not (practically) any improvement of the model. Accuracy: 63%

M3: Using 8Khz data. The idea of the last model was to change a bit the architecture of the model in the downsampled version. In this case we decided to delete one of the dropouts of 0.5 and replacing it with another one of 0.3 but after the first Linear Layer (1664,256) or (128+12x32x3,256). For the moment is the best accuracy that has been obtained. Accuracy: 71%

M3: Using 16Khz data. In this case, we work with all the samples and we put all the dropouts of 0.3 between all the layers but the accuracy doesn't seem to improve. Accuracy: 60%

| Model | 8kHz audio input | 16kHz audio input |
| --- | --- | --- |
| M1    |  64%  |  55%  |
| M2    |  62%  |  63%  |
| M3    |  71%  |  60%  |

More details in [README](https://github.com/unaivicente/FinalProjectAIDL/blob/main/MEL/README_MEL.md)
