# GRU

## DIFFERENT GRU ARCHITECTURES AND HYPERPARAMETER TUNING TESTED

![GRU1](https://user-images.githubusercontent.com/92716609/165112353-5cf541f1-61c5-494e-81fd-eb3a570384d0.png)

* GRU-1 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 128 hidden dimension units that end up to a fully connected linear layer. Negative log likelihood as loss function adding a LogSoftmax layer in the last layer of the network. Trained with 100 of batch size, 10 epochs, 0.5 of learning rate and SGD optimizer without weight decay.
 
 
 
![GRU2](https://user-images.githubusercontent.com/92716609/165112373-fdb0d57f-8bc6-4601-9859-170477e3998a.png)

* GRU-2 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 128 hidden dimension units that end up to a fully connected linear layer. Negative log likelihood as loss function adding a LogSoftmax layer in the last layer of the network. Trained with 50 of batch size, 40 epochs, 0.01 of learning rate and SGD optimizer with 0.0001 weight decay and StepLR that decays de learning rate of each parameter group by 0.1 (gamma) every 20 epochs.


      
![GRU3](https://user-images.githubusercontent.com/92716609/165112397-fffb7195-b528-497b-9662-d99b5862ee67.png)

* GRU-3 : 3 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 256 hidden dimension units that end up to a fully connected linear layer. Cross Entropy Loss as loss function passing the output of the last layer (linear) directly. Trained with 100 of batch size, 10 epochs, 0.5 of learning rate and SGD optimizer without weight decay.

      
      
![GRU4](https://user-images.githubusercontent.com/92716609/165112419-ac0b55d4-af90-451f-8f63-9b09fb00e37d.png)

* GRU-4 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 256 hidden dimension units that end up to a fully connected linear layer. Cross Entropy Loss as loss function passing the output of the last layer (linear) directly. Trained with 100 of batch size, 10 epochs, 1e-4 of learning rate and Adam optimizer.

     
![GRU5](https://user-images.githubusercontent.com/92716609/165112448-caf8afe2-7534-4766-bb76-635424113d95.png)   
     
* GRU-5 : 4 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 128 hidden dimension units that end up to a fully connected linear layer. Negative log likelihood as loss function adding a LogSoftmax layer in the last layer of the network. Trained with 150 of batch size, 10 epochs, 0.01 of learning rate and SGD optimizer without weight decay.


![GRU6](https://user-images.githubusercontent.com/92716609/165112499-9e72b47e-8d79-4a94-afa9-d02b409f193b.png)
 
* GRU-6 : 2 layer 1D convolutional layers for feature extraction that go from 1 to 128 channels with batch normalization and 1D maxpooling and then connecting it to a GRU with 256 hidden dimension units that end up to a fully connected linear layer. Cross Entropy Loss as loss function passing the output of the last layer (linear) directly. Trained with 100 of batch size, 30 epochs, 1e-4 of learning rate and Adam optimizer.


## TRAINING LOSS
To compare the difference between different audio input sample rates in training, we can look at the training loss graphs.

### GRU-1

![gru-1_train_loss](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/training_loss-GRU-1.png)


### GRU-2

![gru-2_train_loss](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/training_loss-GRU-2.png)


### GRU-3

![gru-3_train_loss](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/training_loss-GRU-3.png)


### GRU-4

![gru-4_train_loss](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/training_loss-GRU-4.png)


### GRU-5

![gru-5_train_loss](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/training_loss-GRU-5.png)


### GRU-6

![gru-6_train_loss](https://github.com/unaivicente/FinalProjectAIDL/blob/main/doc/training_loss-GRU-6.png)



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
