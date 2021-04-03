import torch
import torch.nn as nn
import tez
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing, model_selection, metrics
from torch.utils.data import DataLoader


class MovieDataset:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        user = self.users[item]
        movie = self.movies[item]
        rating = self.ratings[item]

        return {"users": torch.tensor(user, dtype=torch.long),
                "movies": torch.tensor(movie, dtype=torch.long),
                "ratings": torch.tensor(rating, dtype=torch.float),
                }

class RecSysModel(tez.Model):
    def __init__(self, num_users, num_movies,
                 optimizerLearningRate = 1e-3, schedulerStepSize = 3,
                 schedulerGamma= 0.5):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, 32)
        self.movie_embed = nn.Embedding(num_movies, 32)
        self.out = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 1))
        self.optimizerLearningRate = optimizerLearningRate
        self.schedulerStepSize = schedulerStepSize
        self.schedulerGamma = schedulerGamma

    def fetch_optimizer(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.optimizerLearningRate)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                              step_size=self.schedulerStepSize, gamma=self.schedulerGamma)
        return sch

    def monitor_metrics(self, output, rating):
        output = output.detach().cpu().numpy()
        rating = rating.detach().cpu().numpy()

        return {
            'rmse': np.sqrt(metrics.mean_squared_error(rating, output))
        }

    def forward(self, users, movies, ratings):
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        output = torch.cat([user_embeds, movie_embeds], dim=1)
        output = self.out(output)

        if len(ratings):
            loss = nn.MSELoss()(output, ratings.view(-1, 1))
            calc_metrics = self.monitor_metrics(output, ratings.view(-1, 1))
            return output, loss, calc_metrics


class Train:

    def __init__(self, trainDatasetPath, userLabelEncoderPath,
                 movieLabelEncoderPath, validDatasetSize,
                 trainBatchSize, validBatchSize,
                 device, nEpochs, trainedModelPath,
                 randomState):

        self.trainDatasetPath = trainDatasetPath
        self.userLabelEncoderPath = userLabelEncoderPath
        self.movieLabelEncoderPath = movieLabelEncoderPath
        self.validDatasetSize = validDatasetSize
        self.randomState = randomState
        self.trainBatchSize = trainBatchSize
        self.validBatchSize = validBatchSize
        self.device = device
        self.nEpochs = nEpochs
        self.trainedModelPath = trainedModelPath

    def train(self):

        self.dataset = pd.read_csv(self.trainDatasetPath)

        self.lbl_user = preprocessing.LabelEncoder()
        self.lbl_movie = preprocessing.LabelEncoder()
        self.lbl_user.fit(self.dataset.userId.values)
        self.lbl_movie.fit(self.dataset.movieId.values)

        output = open(self.userLabelEncoderPath, 'wb')
        pickle.dump(self.lbl_user, output)
        output.close()

        output = open(self.movieLabelEncoderPath, 'wb')
        pickle.dump(self.lbl_movie, output)
        output.close()

        self.dataset.userId = self.lbl_user.transform(
            self.dataset.userId.values)
        self.dataset.movieId = self.lbl_movie.transform(
            self.dataset.movieId.values)

        self.trainDataset, self.validDataset = model_selection.train_test_split(self.dataset, test_size=self.validDatasetSize,
                                                                                random_state=self.randomState,
                                                                                stratify=self.dataset.rating.values)

        self.train_dataset = MovieDataset(users=self.trainDataset.userId.values,
                                          movies=self.trainDataset.movieId.values,
                                          ratings=self.trainDataset.rating.values)

        self.valid_dataset = MovieDataset(users=self.validDataset.userId.values,
                                          movies=self.validDataset.movieId.values,
                                          ratings=self.validDataset.rating.values)

        model = RecSysModel(num_users=len(self.lbl_user.classes_),
                            num_movies=len(self.lbl_movie.classes_))

        model.fit(self.train_dataset, self.valid_dataset, train_bs=self.trainBatchSize,
                  valid_bs=self.validBatchSize, device=self.device, epochs=self.nEpochs)

        model.save(self.trainedModelPath)


class Predict:

    def __init__(self, datasetPath, userLabelEncoderPath, movieLabelEncoderPath,
                 trainedModelPath, predictBatchSize, device=None):

        self.dataset = pd.read_csv(datasetPath)
        self.userLabelEncoderPath = userLabelEncoderPath
        self.movieLabelEncoderPath = movieLabelEncoderPath
        self.trainedModelPath = trainedModelPath
        self.predictBatchSize = predictBatchSize
        self.device = device
        if self.device == None:
            self.device = 'cpu'

    def predict(self):

        pickleFile = open(self.userLabelEncoderPath, 'rb')
        self.lbl_user = pickle.load(pickleFile)
        pickleFile.close()

        pickleFile = open(self.movieLabelEncoderPath, 'rb')
        self.lbl_movie = pickle.load(pickleFile)
        pickleFile.close()

        self.dataset['userId'] = self.lbl_user.transform(
            self.dataset['userId'].values)
        self.dataset['movieId'] = self.lbl_movie.transform(
            self.dataset['movieId'].values)

        self.valid_dataset = MovieDataset(users=self.dataset.userId.values,
                                            movies=self.dataset.movieId.values,
                                            ratings= self.dataset.rating.values)

        self.model = RecSysModel(num_users=len(self.lbl_user.classes_), num_movies = len(self.lbl_movie.classes_))

        self.model.load(self.trainedModelPath, device=self.device)
        self.model.eval()

        self.predicition = self.model.predict(
            self.valid_dataset, batch_size=self.predictBatchSize)
        self.predictedValue = []

        for values in self.predicition:
            for value in values:
                self.predictedValue.append(value.item())

        return self.predictedValue

