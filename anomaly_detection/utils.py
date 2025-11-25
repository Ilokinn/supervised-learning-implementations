from sklearn import metrics as sk_metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score


def plot_loss(train_losses):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker='o', linestyle='-', color='blue',
             label='erreur d\'entraînement (MSE)')
    plt.title('évolution de l\'erreur de reconstruction par époque')
    plt.xlabel('époque')
    plt.ylabel('erreur de reconstruction (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def split_ecg_data_step1(ecg_data):
    """
    Étape 1: Séparer normales et anomalies
    """
    # Séparer les données
    normales = ecg_data[ecg_data[:, -1] == 0]
    anomalies = ecg_data[ecg_data[:, -1] == 1]
    
    print(f"Normales trouvées: {len(normales)}")
    print(f"Anomalies trouvées: {len(anomalies)}")
    
    # Mélanger les données
    normales = normales[np.random.permutation(len(normales))]
    anomalies = anomalies[np.random.permutation(len(anomalies))]
    
    # Diviser NORMALES (60%, 30%, 10%)
    n_train = int(0.6 * len(normales))
    n_test = int(0.3 * len(normales))
    train_normal = normales[:n_train]
    test_normal = normales[n_train:n_train + n_test]
    val_normal = normales[n_train + n_test:]

    # Diviser ANOMALIES (80%, 20%)
    a_test = int(0.8 * len(anomalies))
    test_anomaly = anomalies[:a_test]
    val_anomaly = anomalies[a_test:]
    
    # Composer les ensembles finaux
    train_set = train_normal.copy()
    test_set = np.vstack([test_normal, test_anomaly])
    val_set = np.vstack([val_normal, val_anomaly])
    
    # Mélanger test et val
    test_set = test_set[np.random.permutation(len(test_set))]
    val_set = val_set[np.random.permutation(len(val_set))]
    
    return train_set, test_set, val_set
def split_ecg_data(ecg_data, random_state=42):
    """
    Split les données ECG en TRAIN, VALIDATION, TEST
    Selon l'énoncé: Label 0 = NORMAL, Label 1 = ANOMALIE
    """
    np.random.seed(random_state)
    
    # Selon l'énoncé
    normales = ecg_data[ecg_data[:, -1] == 0]      # Label 0 = normal
    anomalies = ecg_data[ecg_data[:, -1] == 1]     # Label 1 = anomalie
    
    # Mélanger
    normales = normales[np.random.permutation(len(normales))]
    anomalies = anomalies[np.random.permutation(len(anomalies))]
    
    # Split NORMALES (60%, 30%, 10%)
    n_train_normal = int(0.60 * len(normales))
    n_val_normal = int(0.10 * len(normales))
    
    train_normal = normales[:n_train_normal]
    val_normal = normales[n_train_normal:n_train_normal + n_val_normal]
    test_normal = normales[n_train_normal + n_val_normal:]
    
    # Split ANOMALIES (0%, 20%, 80%)
    n_val_anomaly = int(0.20 * len(anomalies))
    
    val_anomaly = anomalies[:n_val_anomaly]
    test_anomaly = anomalies[n_val_anomaly:]
    
    # Composer les ensembles
    train_set = train_normal # sortie 0
    
    val_set = np.vstack([val_normal, val_anomaly])
    val_set = val_set[np.random.permutation(len(val_set))] # sortie 1
    
    test_set = np.vstack([test_normal, test_anomaly])
    test_set = test_set[np.random.permutation(len(test_set))] # sortie 2
    
    return train_set, val_set, test_set
def data_split(path):
    # read file npz and doing the split
    data = numpy.load(path)
    normal_data = data[data[:, -1] == 0]
    anomaly_data = data[data[:, -1] == 1]
    print(normal_data.shape, np.min(normal_data), np.max(normal_data))
    print(anomaly_data.shape, np.min(anomaly_data), np.max(anomaly_data))

    normal_data = normal_data.sample(frac=1, random_state=42).reset_index(drop=True)
    anomaly_data = anomaly_data.sample(frac=1, random_state=42).reset_index(drop=True)

    n_train = int(0.6 * len(normal_data))
    n_test = int(0.3 * len(normal_data))
    train_normal = normal_data[:n_train]
    test_normal = normal_data[n_train:n_train + n_test]
    val_normal = normal_data[n_train + n_test:]

    a_test = int(0.8 * len(anomaly_data))
    test_anomaly = anomaly_data[:a_test]
    val_anomaly = anomaly_data[a_test:]

    train_set = train_normal.copy()
    test_set = pd.concat([test_normal, test_anomaly], ignore_index=True)
    val_set = pd.concat([val_normal, val_anomaly], ignore_index=True)

    test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)
    val_set = val_set.sample(frac=1, random_state=42).reset_index(drop=True)
    return train_set.values[:, :-1], test_set.values, val_set.values

def compute_metrics(val_score, y_val, thresh, pos_label=1):
    y_pred = (val_score >= thresh).astype(int)
    y_true = y_val.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    avgpr = sk_metrics.average_precision_score(y_true, val_score)
    roc = sk_metrics.roc_auc_score(y_true, val_score)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

    return accuracy, precision, recall, f_score, roc, avgpr, cm

def compute_metrics_binary(y_pred, y_val, pos_label=1):
    y_true = y_val.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])
    return {
        "acc": accuracy,
        "prec": precision,
        "rec": recall,
        "f1": f_score,
        "cm": cm
    }
def compute_anomaly_metrics(y_pred, y_test, pos_label=1):
    """
    Calcule les métriques pour anomaly detection
    """
    y_true = y_test.astype(int)
    
    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    roc_auc = sk_metrics.roc_auc_score(y_true, y_pred)
    cm = sk_metrics.confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f_score,
        "roc_auc": roc_auc,
        "cm": cm
    }

def estimate_optimal_threshold(val_score, y_val, pos_label=1, nq=100):
    ratio = 100 * sum(y_val == 0) / len(y_val)
    q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
    thresholds = np.percentile(val_score, q)

    result_search = []
    confusion_matrices = []
    f1 = np.zeros(shape=nq)
    r = np.zeros(shape=nq)
    p = np.zeros(shape=nq)
    auc = np.zeros(shape=nq)
    aupr = np.zeros(shape=nq)
    qis = np.zeros(shape=nq)

    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        metrics = compute_metrics(val_score, y_val, thresh, pos_label)
        accuracy = metrics["acc"]
        precision = metrics['prec']
        recall = metrics['rec']
        f_score = metrics['f1']
        roc = metrics['roc']
        avgpr = metrics['avgpr']
        cm = metrics['cm']

        confusion_matrices.append(cm)
        result_search.append([accuracy, precision, recall, f_score])
        f1[i] = f_score
        r[i] = recall
        p[i] = precision
        auc[i] = roc
        aupr[i] = avgpr
        qis[i] = qi

    arm = np.argmax(f1)

    return thresholds[arm]


def compute_metrics(test_score, y_test, thresh, pos_label=1):
    y_pred = (test_score >= thresh).astype(int)
    y_true = y_test.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    avgpr = sk_metrics.average_precision_score(y_true, test_score)
    roc = sk_metrics.roc_auc_score(y_true, test_score, max_fpr=1e-2)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])
    return {
        "acc": accuracy,
        "prec": precision,
        "rec": recall,
        "f1": f_score,
        "roc": roc,
        "avgpr": avgpr,
        "cm": cm
    }





def clustering_metrics(DATA,LABELS_PREDICT, LABELS_TRUE) :
#     | Métrique      | Pourquoi elle est importante |
#     | ------------- | ---------------------------- |
#     | **Recall**    | détecter les anomalies       |
#     | **Precision** | éviter faux positifs         |
#     | **F1**        | équilibre global             |
#     | **AUC**       | puissance du modèle          |

    # data (either your original data, or the reduced data from UMAP/PCA)

    # Les metrics pour acp vs umap 
    # Silhouette - 

    # Silhoutte 
    silh_score = silhouette_score(DATA,LABELS_PREDICT,random_state=42) 

    # Davies Bouldin
    db_score = davies_bouldin_score(DATA,labels=LABELS_PREDICT)
    
    # Calinski Harabasz
    ch_score = calinski_harabasz_score(DATA,labels=LABELS_PREDICT)
    
    # ARI 
    ari_score = adjusted_rand_score(labels_true=LABELS_TRUE, labels_pred=LABELS_PREDICT)

    # Nmi 
    
    nmi_score = normalized_mutual_info_score(labels_true=LABELS_TRUE, labels_pred=LABELS_PREDICT)

    results = {
          "Silhouette Score": silh_score,
          "Davies Bouldin Score": db_score,
          "Calinski Harabasz Score": ch_score,
          "Adjusted Rand Index (ARI)": ari_score,
          "Normalized Mutual Info (NMI)": nmi_score
               } 
    return results


def run_clustering_multiple_seeds(X, algo_model_func, algo_name, n_clusters=3, n_seeds=10,y_true=None, **algo_params):
    # Exécute un algorithme de clustering avec plusieurs seeds et calcule μ ± σ
    results_dict = {
        'labels_list': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
        'ari': [],
        'nmi': []
    }
    # ========== K-MEANS ==========
    if algo_name.lower() == 'kmeans':
        for seed in range(n_seeds):
            model = algo_model_func(N_clusters=n_clusters, random_state=seed,)
            labels = model.fit_predict(X)
            results_dict['labels_list'].append(labels)
            
            # Métriques internes
            results_dict['silhouette'].append(silhouette_score(X, labels))
            results_dict['davies_bouldin'].append(davies_bouldin_score(X, labels))
            results_dict['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
            
            # Métriques externes (si labels vrais)
            if y_true is not None:
                results_dict['ari'].append(adjusted_rand_score(y_true, labels))
                results_dict['nmi'].append(normalized_mutual_info_score(y_true, labels))
    # ========== DBSCAN ==========
    elif algo_name.lower() == 'dbscan':
        eps = algo_params.get('eps', 0.5)
        min_samples = algo_params.get('min_samples', 5)
        
        for seed in range(n_seeds):
            model = algo_model_func(MIN_SAMPLES=min_samples, EPS=eps)
            labels = model.fit_predict(X)
            results_dict['labels_list'].append(labels)
            
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Silhouette seulement si pas trop de bruit
            if n_clusters_found > 1 and -1 not in labels:
                results_dict['silhouette'].append(silhouette_score(X, labels))
                results_dict['davies_bouldin'].append(davies_bouldin_score(X, labels))
                results_dict['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
            else:
                results_dict['silhouette'].append(np.nan)
                results_dict['davies_bouldin'].append(np.nan)
                results_dict['calinski_harabasz'].append(np.nan)
            
            if y_true is not None:
                results_dict['ari'].append(adjusted_rand_score(y_true, labels))
                results_dict['nmi'].append(normalized_mutual_info_score(y_true, labels))
    
    # ========== SPECTRAL ==========
    elif algo_name.lower() == 'spectral':
        for seed in range(n_seeds):
            model = algo_model_func(N_clusters=n_clusters, Random_State=seed)
            labels = model.fit_predict(X)
            results_dict['labels_list'].append(labels)
            
            results_dict['silhouette'].append(silhouette_score(X, labels))
            results_dict['davies_bouldin'].append(davies_bouldin_score(X, labels))
            results_dict['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
            
            if y_true is not None:
                results_dict['ari'].append(adjusted_rand_score(y_true, labels))
                results_dict['nmi'].append(normalized_mutual_info_score(y_true, labels))

    
    return results_dict


def plot_2d_embedding(X_2d, labels, true_labels=None, title="", save_path=None):
    """
    Crée un scatter plot 2D montrant les clusters prédits et vrais.
    
    Paramètres :
    -----------
    X_2d : array-like, shape (n_samples, 2)
        Données réduites à 2 dimensions (PCA 2D ou UMAP 2D)
    
    labels : array-like, shape (n_samples,)
        Labels prédits par l'algorithme
    
    true_labels : array-like, shape (n_samples,), optionnel
        Labels vrais (pour comparaison side-by-side)
    
    title : str, default=""
        Titre du graphique
    
    save_path : str, optionnel
        Chemin pour sauvegarder l'image (ex: "plots/kmeans_pca.png")
    
    Retour :
    --------
    Affiche le graphique et le sauvegarde si save_path est fourni
    """
    
    # ════ ÉTAPE 1 : Décider le nombre de subplots ════
    if true_labels is not None:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax = [ax]  # Convertir en liste pour cohérence
    
    # ════ ÉTAPE 2 : Subplot 1 - Labels PRÉDITS ════
    scatter1 = ax[0].scatter(X_2d[:, 0], X_2d[:, 1], 
                             c=labels, 
                             cmap='viridis', 
                             s=30, 
                             alpha=0.7, 
                             edgecolors='k', 
                             linewidth=0.5)
    
    ax[0].set_xlabel('Dimension 1 (PC1 ou UMAP1)', fontsize=11)
    ax[0].set_ylabel('Dimension 2 (PC2 ou UMAP2)', fontsize=11)
    ax[0].set_title(f'{title} - Labels Prédits', fontsize=12, fontweight='bold')
    plt.colorbar(scatter1, ax=ax[0], label='Cluster')
    
    # ════ ÉTAPE 3 : Subplot 2 - Labels VRAIS (optionnel) ════
    if true_labels is not None:
        scatter2 = ax[1].scatter(X_2d[:, 0], X_2d[:, 1], 
                                 c=true_labels, 
                                 cmap='tab10', 
                                 s=30, 
                                 alpha=0.7, 
                                 edgecolors='k', 
                                 linewidth=0.5)
        
        ax[1].set_xlabel('Dimension 1 (PC1 ou UMAP1)', fontsize=11)
        ax[1].set_ylabel('Dimension 2 (PC2 ou UMAP2)', fontsize=11)
        ax[1].set_title(f'{title} - Labels Vrais', fontsize=12, fontweight='bold')
        plt.colorbar(scatter2, ax=ax[1], label='Classe')
    
    # ════ ÉTAPE 4 : Finaliser ════
    plt.tight_layout()
    
    # ════ ÉTAPE 5 : Sauvegarder (optionnel) ════
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé : {save_path}")
    
    # ════ ÉTAPE 6 : Afficher ════
    plt.show()


# ============================================================
# ANOMALY DETECTION - AUTO-ENCODEUR
# ============================================================

# def estimate_optimal_threshold(val_scores, y_val, pos_label=1, nq=100):
#     """
#     Trouve le meilleur seuil en testant différentes valeurs
#     Maximise le F1-score sur les données de validation
#     """
#     # Ratio de normales dans la validation
#     ratio = 100 * sum(y_val == 0) / len(y_val)
    
#     # Test des seuils autour de ce ratio
#     q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
#     thresholds = np.percentile(val_scores, q)
    
#     f1_scores = np.zeros(nq)
    
#     for i, thresh in enumerate(thresholds):
#         y_pred = (val_scores >= thresh).astype(int)
#         y_true = y_val.astype(int)
        
#         precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
#             y_true, y_pred, average='binary', pos_label=pos_label
#         )
#         f1_scores[i] = f_score
    
#     # Retourner le seuil avec le meilleur F1
#     best_idx = np.argmax(f1_scores)
#     return thresholds[best_idx], f1_scores[best_idx]


# def compute_ae_metrics(test_scores, y_test, thresh, pos_label=1):
#     """
#     Calcule les métriques pour Auto-encodeur
#     """
#     y_pred = (test_scores >= thresh).astype(int)
#     y_true = y_test.astype(int)
    
#     accuracy = sk_metrics.accuracy_score(y_true, y_pred)
#     precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
#         y_true, y_pred, average='binary', pos_label=pos_label
#     )
#     roc_auc = sk_metrics.roc_auc_score(y_true, test_scores)
#     avgpr = sk_metrics.average_precision_score(y_true, test_scores)
#     cm = sk_metrics.confusion_matrix(y_true, y_pred)
    
#     return {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f_score,
#         "roc_auc": roc_auc,
#         "avgpr": avgpr,
#         "cm": cm
#     }