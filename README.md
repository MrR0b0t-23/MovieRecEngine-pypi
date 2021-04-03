# MovieRecEngine

MovieRecEngine be an abbreviation of Movie Recommendation Engine. This is a simple collaborative filtering based library using Pytorch Neural Netwrok to make your Movie Recommendation System easy.

*This library is in very early-stage currently! So, there might be remarkable changes.*

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install MovieRecEngine.

```bash
pip install MovieRecEngine
```
## Description

MovieRecEngine uses collaborative filtering to find similarities between users and items simultaneously to provide recommendations. This allows for serendipitous recommendations; that is, collaborative filtering models can recommend an item to user A based on the interests of a similar user B. Furthermore, the embeddings can be learned automatically, without relying on hand-engineering of features. 

MovieRecEngine, uses [tez](https://pypi.org/project/tez/) simple pytorch trainer that supports cpu and gpu training.

## How to use MovieRecEngine

* To train a model using MovieRecEngine, define a Dataset that contains columns "userId", "movieId", "ratings". Example [MovieLenz](https://grouplens.org/datasets/movielens/100k/) dataset.
* Create a object for ```python Train ``` class in MovieRecEngine library with parameters trainDatasetPath, userLabelEncoderPath, movieLabelEncoderPath, validDatasetSize, trainBatchSize, validBatchSize, device, nEpochs, trainedModelPath, randomState.
* Train the model by calling ```python train``` function in ```python Train``` class.

* To predict user movie ratings using MovieRecEngine, define a Dataset that contains columns "userId", "movieId", "ratings". 
*NOTE: "userId" needs to contain 1 unique userId.*
* Create a object for ```python Predict ``` class in MovieRecEngine library with parameters datasetPath, userLabelEncoderPath, movieLabelEncoderPath, trainedModelPath, predictBatchSize, device.
* Predict user movie ratings by calling ```python predict``` function in ```python Predict ``` class.

## Parameters


## Contributing

Currently, we are not accepting any pull requests! All PRs will be closed. If you want a feature or something doesn't work, please create an [issue](https://github.com/MrR0b0t-23/MovieRecEngine/issues).

## License
