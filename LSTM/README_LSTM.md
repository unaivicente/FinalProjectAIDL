Here we present the following model architectures:


![L1](https://user-images.githubusercontent.com/92716609/165109468-1648f7c6-7543-44de-a3e4-1e6310ec8f86.png)

L1: Using 16Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling and then connecting it to a LSTM with 128 hidden dimension units that end up to a fully connected linear layer.

![L1_16](https://user-images.githubusercontent.com/92716609/165107940-2f3a0b5f-8f0e-45db-bfdd-8eabcafc9de7.png)
  * Accuracy: 92%


![L2](https://user-images.githubusercontent.com/92716609/165109483-2ff8e910-d588-433a-ab29-9bc5f2741bd0.png)

L2: Using 16Khz data. Same as L1 but with 256 hidden layers in the LSTM layer.

![L2_16](https://user-images.githubusercontent.com/92716609/165108084-7c0c01e2-7c23-4798-b6f3-c217b0601c14.png)
  * Accuracy: 92%


![L3](https://user-images.githubusercontent.com/92716609/165109514-73e9353b-2531-49e9-99b9-4ba7b0235d3f.png)

L3: Using 16Khz data. 3 layer 1D convolutional layers in the same sequence as in L1.

![L3_16](https://user-images.githubusercontent.com/92716609/165108235-ef43bdfe-6d71-42cb-af52-7c4e87c8ea6a.png)
  * Accuracy: 88%


![L4](https://user-images.githubusercontent.com/92716609/165109544-d5fe7fb5-a42c-42f0-86b6-4214b0fdbf05.png)

L4: Using 16Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling, we add another 5th layer with maxpooling and then connect 1024 channels to a LSTM with 128 hidden dimension units that end up to a fully connected linear layer.

![L4_16](https://user-images.githubusercontent.com/92716609/165108381-b54149e4-6b4b-4804-af76-ef50f948a916.png)
  * Accuracy: 91%


![L5](https://user-images.githubusercontent.com/92716609/165109560-6c4066cd-afe0-4009-b1d6-0a259977a666.png)

L5: Using 16Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling, we add another two layers (5th and 6th) with maxpooling and then connect 2048 channels to a LSTM with 256 hidden dimension units that end up to a fully connected linear layer.

![L5_16](https://user-images.githubusercontent.com/92716609/165108398-8ef13283-d1d6-4106-9f4e-52bc37ef6aaa.png)
  * Accuracy: 91%

L1: Using 8Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling and then connecting it to a LSTM with 128 hidden dimension units that end up to a fully connected linear layer.

![L1_8](https://user-images.githubusercontent.com/92716609/165107984-1f7c7276-9747-42c5-af3f-e282f8681305.png)
  * Accuracy: 90%

L2: Using 8Khz data. Same as L1 but with 256 hidden layers in the LSTM layer.

![L2_8](https://user-images.githubusercontent.com/92716609/165108138-3487e371-f074-4f54-9743-32140254dcbb.png)
  * Accuracy: 90%

L3: Using 8Khz data. 3 layer 1D convolutional layers in the same sequence as in L1.

![L3_8](https://user-images.githubusercontent.com/92716609/165108276-ad8f3eae-1145-45f7-a674-b0c3f90abf00.png)
  * Accuracy: 88%

L4: Using 8Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling, we add another 5th layer with maxpooling and then connect 1024 channels to a LSTM with 128 hidden dimension units that end up to a fully connected linear layer.

![L4_8](https://user-images.githubusercontent.com/92716609/165108318-16727ccb-73bf-4a14-89eb-f3204b79ec57.png)
  * Accuracy: 79%

L5: Using 8Khz. We start with 4 layer 1D convolutional layers for feature extraction that go from 1 to 256 channels with batch normalization and 1D maxpooling, we add another two layers (5th and 6th) with maxpooling and then connect 2048 channels to a LSTM with 256 hidden dimension units that end up to a fully connected linear layer.

![L5_8](https://user-images.githubusercontent.com/92716609/165108340-f22028f1-a4a6-4e7b-82ef-e76722a31252.png)
  * Accuracy: 88%
