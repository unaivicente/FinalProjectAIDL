
Here we present the following model architectures:

C1: 4 layer 1D convolutional layers (Conv1d) that go from 1 to 64 channels, with batch normalization and 1D maxpooling (Maxpool1d). It is the reference and it is directly extracted from M5 model in https://arxiv.org/pdf/1610.00087.pdf
  * Accuracy: 83%
C2: 6 layer 1D convolutional layers with extended channels from 1 to 2048, also same batch normalization and 1D maxpooling until layer 4.
  * Accuracy: 
