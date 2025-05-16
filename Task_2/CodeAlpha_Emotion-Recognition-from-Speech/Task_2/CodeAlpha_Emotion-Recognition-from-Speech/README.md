# CodeAlpha_Emotion-Recognition-from-Speech
Built a model that can recognize emotions in speech audio. Used deep learning and speech processing techniques to classify spoken sentences into different emotions like happiness, anger, or sadness.

# Dataset Information

Unfortunately, the dataset is too big to upload on github and thus I have provided the link to download it below.

**Download link:** https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess

There are a set of 200 target words were spoken in the carrier phrase "Say the word _' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.

The dataset is organised such that each of the two female actor and their emotions are contain within its own folder. And within that, all 200 target words audio file can be found. The format of the audio file is a WAV format

### Output Attributes
- anger
- disgust
- fear
- happiness
- pleasant surprise
- sadness
- neutral

# Libraries

- pandas
- matplotlib
- keras
- tensorflow
- librosa

# Neural Network

<li>LSTM Network
  
**Accuracy:** 67.00
