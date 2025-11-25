
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

import numpy as np
import matplotlib.pyplot as plt

class IF:
    def __init__(self, args):
        self.clf = IsolationForest(n_estimators=args.n_estimators, random_state=42)
        self.name = "IF"

    def get_params(self) -> dict:
        return {
            "n_estimators": self.clf.n_estimators
        }

class GMM:
    def __init__(self, args):
        self.clf = GaussianMixture(n_components=args.n_components, covariance_type='full', random_state=42)
        self.name = "GMM"

    def get_params(self) -> dict:
        return {
            "n_components": self.clf.n_components
        }

class AE(nn.Module):
    def __init__(self, in_features):
        super(AE, self).__init__()
        self.name = 'AE'
        self.enc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.dec = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, in_features),
        )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode

class MCDropoutAE(nn.Module):
    def __init__(self, in_features, dropout_rate=0.5):
        super(MCDropoutAE, self).__init__()
        self.name = 'MCDropoutAE'
        self.dropout_rate = dropout_rate
        
        self.enc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 8)
        )
        
        self.dec = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, in_features)
        )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode

def get_kmeans_model(N_clusters,random_state):
    kmeans_model = KMeans(n_clusters=N_clusters,random_state=random_state,n_init=10,max_iter=300)
    return kmeans_model

def fit_kmeans(Model,X):
    Model.fit(X)
    labels = Model.labels_
    return labels

def fit_dbscan(Model, X):
    Model.fit(X)
    labels = Model.labels_
    return labels
def fit_spectral_clustering(Model, X):

    labels = Model.fit_predict(X)
    return labels

def get_dbscan_model(MIN_SAMPLES=5,EPS=0.5):
    db_scan_model = DBSCAN(min_samples=MIN_SAMPLES,eps=EPS)
    return db_scan_model

def get_spectral_model(N_clusters,Random_State):
    spectral_model= SpectralClustering(n_clusters=N_clusters, random_state=Random_State)
    return spectral_model

