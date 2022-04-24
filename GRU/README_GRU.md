# GRU

## DIFFERENT GRU ARCHITECTURES AND HYPERPARAMETER TUNING TESTED

![gru1_diagram](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/gru-1_diagram.png)


* GRU-1 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 128 hidden dimension units that end up to a fully connected linear layer. Negative log likelihood as loss function adding a LogSoftmax layer in the last layer of the network. Trained with 100 of batch size, 10 epochs, 0.5 of learning rate and SGD optimizer without weight decay.
 
 
 
 ![gru2_diagram](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/gru-2_diagram.png)

* GRU-2 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 128 hidden dimension units that end up to a fully connected linear layer. Negative log likelihood as loss function adding a LogSoftmax layer in the last layer of the network. Trained with 50 of batch size, 40 epochs, 0.01 of learning rate and SGD optimizer with 0.0001 weight decay and StepLR that decays de learning rate of each parameter group by 0.1 (gamma) every 20 epochs.


      
 ![gru3_diagram](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/gru-3_diagram.png)

* GRU-3 : 3 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 256 hidden dimension units that end up to a fully connected linear layer. Cross Entropy Loss as loss function passing the output of the last layer (linear) directly. Trained with 100 of batch size, 10 epochs, 0.5 of learning rate and SGD optimizer without weight decay.

      
      
![gru4_diagram](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/gru-4_diagram.png)

* GRU-4 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 256 hidden dimension units that end up to a fully connected linear layer. Cross Entropy Loss as loss function passing the output of the last layer (linear) directly. Trained with 100 of batch size, 10 epochs, 1e-4 of learning rate and Adam optimizer.

     
     
![gru5_diagram](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/gru-5_diagram.png)
     
* GRU-5 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 128 hidden dimension units that end up to a fully connected linear layer. Negative log likelihood as loss function adding a LogSoftmax layer in the last layer of the network. Trained with 150 of batch size, 10 epochs, 0.01 of learning rate and SGD optimizer without weight decay.

 
 
 ![gru6_diagram](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/gru-6_diagram.png)
 
* GRU-6 : 2 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 256 hidden dimension units that end up to a fully connected linear layer. Cross Entropy Loss as loss function passing the output of the last layer (linear) directly. Trained with 100 of batch size, 30 epochs, 1e-4 of learning rate and Adam optimizer.


## TRAINING LOSS
To compare the difference between different audio input sample rates in training, we can look at the training loss graphs.

### GRU-1



### GRU-2

### GRU-3

### GRU-4

### GRU-5

### GRU-6


## ACCURACY RESULTS

The results of accuracy with the different sample rate of the audio input:
      
| Model | Accuracy 8kHz audio input | Accuracy 16kHz audio input |
| --- | --- | --- |
| GRU-1     |  90%   |  92%  |
| GRU-2     |  90%   |  91%  |
| GRU-3     |  92%   |  90%  |
| GRU-4     |  89%   |  91%  |
| GRU-5     |  73%   |  64%  |
| GRU-6     |  85%   |  85%  |
