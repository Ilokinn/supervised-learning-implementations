import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import umap
import numpy
pd.set_option('display.max_columns', 30)

def load_hiseq_data(data_path, labels_path):
    data = pd.read_csv(data_path, index_col=0) 
    labels = pd.read_csv(labels_path, names=["Sample", "Class"], skiprows=1)

    if data.isnull().sum().sum() > 0:
        data = data.fillna(data.mean())

    if "Class" in labels.columns:
        y = labels["Class"]
    else:
        y = labels.iloc[:, -1]
        
    X = data.to_numpy(dtype=float) 
    return X, y

def apply_pca(X,Nb=100):
   pca = PCA(n_components=Nb)
   x_PCA = pca.fit_transform(X)
   return x_PCA

def apply_umap(X,Nb=100):
   umap_model  = umap.UMAP(n_components=Nb,random_state=42)
   X_umap  = umap_model.fit_transform(X)   
   return X_umap

def normalize_data(X):
   scaler = StandardScaler()
   x_scaled = scaler.fit_transform(X)
   return x_scaled

def preprocess_hiseq(X):
    X_scaled = normalize_data(X)

    x_PCA = apply_pca(X_scaled,Nb=100)

    X_umap  = apply_umap(X_scaled,100)

    return X_scaled, x_PCA, X_umap 



def scan_fichier_npz(path_fichier):
    data = numpy.load(path_fichier)
    return data


