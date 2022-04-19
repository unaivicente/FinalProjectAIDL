
Here we present the following model architectures based only on 1D Convolutional Layers:

C1: Using 8Khz data. 4 layer 1D convolutional layers (Conv1d) that go from 1 to 64 channels, with batch normalization and 1D maxpooling (Maxpool1d). It is the reference and it is directly extracted from M5 model in https://arxiv.org/pdf/1610.00087.pdf
  * Accuracy: 77%

C2: Using 16Khz data. 4 layer 1D convolutional layers (Conv1d) that go from 1 to 64 channels, with batch normalization and 1D maxpooling (Maxpool1d). We use the exact same M5 architecture as before and we decide to stay with 16Khz data as it seems like can have a better feature extraction.
  * Accuracy: 83%

C3: Using 8Khz data. 6 layer 1D convolutional layers with extended channels from 1 to 2048, also same batch normalization and 1D maxpooling until layer 4.
  * Accuracy: 

C4: Using 16Khz data. 6 layer 1D convolutional layers with extended channels from 1 to 2048, also same batch normalization and 1D maxpooling until layer 4.
  * Accuracy: 85%
