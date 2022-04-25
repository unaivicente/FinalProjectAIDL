
Here we present the following model architectures based only on 1D Convolutional Layers:

C1: Using 8Khz data. 4 layer 1D convolutional layers (Conv1d) that go from 1 to 64 channels, with batch normalization and 1D maxpooling (Maxpool1d). It is the reference and it is directly extracted from M5 model in https://arxiv.org/pdf/1610.00087.pdf

  * Accuracy: 77%
![C1_8](https://user-images.githubusercontent.com/92716609/165107022-abf27917-4df1-42d5-b5e6-590063fc367d.png)

C1: Using 16Khz data. 4 layer 1D convolutional layers (Conv1d) that go from 1 to 64 channels, with batch normalization and 1D maxpooling (Maxpool1d). We use the exact same M5 architecture as before and we decide to stay with 16Khz data as it seems like can have a better feature extraction.

  * Accuracy: 83%
![C1_16](https://user-images.githubusercontent.com/92716609/165107047-2610ce32-bb3b-4231-8724-3b9097480547.png)

C2: Using 8Khz data. 6 layer 1D convolutional layers with extended channels from 1 to 2048, also same batch normalization and 1D maxpooling until layer 4.

  * Accuracy: 80%
![C2_8](https://user-images.githubusercontent.com/92716609/165107061-ffc2d5de-db6d-4c65-b58e-e0e95d7db333.png)

C2: Using 16Khz data. 6 layer 1D convolutional layers with extended channels from 1 to 2048, also same batch normalization and 1D maxpooling until layer 4.

  * Accuracy: 85%
![C2_16](https://user-images.githubusercontent.com/92716609/165107137-fde20a94-874c-44b6-ad5a-516ce6217461.png)
