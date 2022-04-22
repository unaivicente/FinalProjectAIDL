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


![cnngraph](https://user-images.githubusercontent.com/92716609/164751341-ef5ebeb8-6da2-4533-b50d-871e4ec74019.png)
![lstmgraph](https://user-images.githubusercontent.com/92716609/164751348-10fe9c45-efdb-464f-b99e-46fe77e1f1cc.png)
![grugraph](https://user-images.githubusercontent.com/92716609/164751352-881c90e6-e827-403c-9d3c-07bea17950a6.png)

