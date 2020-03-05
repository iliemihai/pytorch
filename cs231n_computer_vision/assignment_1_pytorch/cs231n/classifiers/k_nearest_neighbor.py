import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

class KNearestNeighbor(nn.Module):
    def __init__(self):
        super(KNearestNeighbor, self).__init__()

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)

        return self.predict_labels(dists, k=k)
    
    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = torch.zeros((num_test, num_train))

        test_square = torch.sum(torch.square(X), axis=1).view(num_test, 1)
        train_square = torch.sum(torch.square(self.X_train), axis=1).view(1, num_train)
        dists = torch.sqrt(test_square + train_square - 2*torch.dot(X, self.X_train.T))

        return dists

    def most_frq_smaller(self, l):
        occ = Counter(l)
        return occ.most_common(1)[0][0]

    def predict_labels(self, dists, k=1):

        num_test = X.shape[0]
        y_pred = torch.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = self.y_train[torch.argsort(dists[i, :])[:k]].tolist()
            y_pred[i] = self.most_frq_smaller(closest_y)