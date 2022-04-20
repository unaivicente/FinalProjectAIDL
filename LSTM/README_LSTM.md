Here we present the following model architectures:

L1: Using 16Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling and then connecting it to a LSTM with 128 hidden dimension units that end up to a fully connected linear layer.
  * Accuracy: 92%

L2: Using 16Khz data. Same as L1 but with 256 hidden layers in the LSTM layer.
  * Accuracy: 92%

L3: Using 16Khz data. 3 layer 1D convolutional layers in the same sequence as in L1.
  * Accuracy: 88%

L4: Using 16Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling, we add another 5th layer with maxpooling and then connect 1024 channels to a LSTM with 128 hidden dimension units that end up to a fully connected linear layer.
  * Accuracy: 91%

L5: Using 16Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling, we add another two layers (5th and 6th) with maxpooling and then connect 2048 channels to a LSTM with 256 hidden dimension units that end up to a fully connected linear layer.
  * Accuracy: 91%

L6: Using 8Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling and then connecting it to a LSTM with 128 hidden dimension units that end up to a fully connected linear layer.
  * Accuracy: 90%

L7: Using 8Khz data. Same as L1 but with 256 hidden layers in the LSTM layer.
  * Accuracy: 90%

L8: Using 8Khz data. 3 layer 1D convolutional layers in the same sequence as in L1.
  * Accuracy: 88%

L9: Using 8Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling, we add another 5th layer with maxpooling and then connect 1024 channels to a LSTM with 128 hidden dimension units that end up to a fully connected linear layer.
  * Accuracy: %

L10: Using 8Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling, we add another two layers (5th and 6th) with maxpooling and then connect 2048 channels to a LSTM with 256 hidden dimension units that end up to a fully connected linear layer.
  * Accuracy: %
