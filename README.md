# MovieRecEngine

MovieRecEngine be an abbreviation of Movie Recommendation Engine. This is a simple collaborative filtering based library using Pytorch Sequential Neural Network to make your Movie Recommendation System easy.

*This library is in very early-stage currently! So, there might be remarkable changes.*

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install MovieRecEngine.

```bash
pip install MovieRecEngine
```
## Description

MovieRecEngine uses collaborative filtering to find similarities between users and items simultaneously to provide recommendations. This allows for serendipitous recommendations; that is, collaborative filtering models can recommend an item to user A based on the interests of a similar user B. Furthermore, the embeddings can be learned automatically, without relying on hand-engineering of features. 

MovieRecEngine uses pyptorch sequential Neural Networks to train a model that can predict users rating for an unseen movie based on his/her past interests/ratings provided. 

MovieRecEngine, uses [tez](https://pypi.org/project/tez/) simple pytorch trainer that supports cpu and gpu training.

## How to use MovieRecEngine

* To train a model using MovieRecEngine, define a Dataset that contains columns "userId", "movieId", "ratings". Example [Train sample](https://github.com/MrR0b0t-23/MovieRecEngine/blob/main/Examples/Train_Sample.csv)
* Create a object for ```Train ``` class in MovieRecEngine library with parameters trainDatasetPath, userLabelEncoderPath, movieLabelEncoderPath, validDatasetSize, trainBatchSize, validBatchSize, device, nEpochs, trainedModelPath, randomState.
* Train the model by calling ```train``` function in ```Train``` class.

* To predict user movie ratings using MovieRecEngine, define a Dataset that contains columns "userId", "movieId", "ratings". Example [Predict sample](https://github.com/MrR0b0t-23/MovieRecEngine/blob/main/Examples/Predict_Sample.csv)

*NOTE: "userId" needs to contain 1 unique userId.*
* Create a object for ```Predict ``` class in MovieRecEngine library with parameters datasetPath, userLabelEncoderPath, movieLabelEncoderPath, trainedModelPath, predictBatchSize, device.
* Predict user movie ratings by calling ```predict``` function in ```Predict ``` class.

## Parameters

1. ```Train``` class: 
- trainDatasetPath ==> Path for your training Dataset.
- userLabelEncoderPath ==> Path in which you want to save user Label Encoder (this will be used in your prediction)
- movieLabelEncoderPath ==> Path in which you want to save movie Label Encoder (this will be used your prediction)
- validDatasetSize ==> Test size for train_test_split 
- trainBatchSize ==> The number of train samples to work through before updating the internal model parameters.
- validBatchSize ==> The number of test samples to work through before updating the internal model parameters.
- device ==> Device in which you want to train your model 'cuda' or 'cpu'. Default 'cpu'. 
- nEpochs ==> The number times that the learning algorithm will work through the entire training dataset.
- trainedModelPath ==> Path to save your trained model (this will be used in your prediction)
- randomState ==> Random State values for train_test_split

2. ```Predict``` class:

- datasetPath ==> Path for your prediction Dataset.
- userLabelEncoderPath ==> Path in which you saved user Label Encoder (while training)
- movieLabelEncoderPath ==>  Path in which you saved movie Label Encoder (while training)
- trainedModelPath ==>  Path in which you saved Trained model (while training)
- predictBatchSize ==> The number of prediction samples to work
- device ==> Device in which you want to train your model 'cuda' or 'cpu'. Default 'cpu'.

## Contributing

Currently, we are not accepting any pull requests! All PRs will be closed. If you want a feature or something doesn't work, please create an [issue](https://github.com/MrR0b0t-23/MovieRecEngine/issues).
