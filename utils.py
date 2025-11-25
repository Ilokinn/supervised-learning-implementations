import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as sk_metrics
from sklearn.metrics import (silhouette_score, davies_bouldin_score, 
                             calinski_harabasz_score, adjusted_rand_score, 
                             normalized_mutual_info_score)

def split_ecg_data(ecg_data, seed=42):
    np.random.seed(seed)
    
    normales = ecg_data[ecg_data[:, -1] == 0]
    anomalies = ecg_data[ecg_data[:, -1] == 1]
    
    normales = normales[np.random.permutation(len(normales))]
    anomalies = anomalies[np.random.permutation(len(anomalies))]
    
    n_total_norm = len(normales)
    n_train_norm = int(0.60 * n_total_norm)
    n_val_norm = int(0.10 * n_total_norm)
    
    train_normal = normales[:n_train_norm]
    val_normal = normales[n_train_norm:n_train_norm + n_val_norm]
    test_normal = normales[n_train_norm + n_val_norm:]
    
    n_total_anom = len(anomalies)
    n_val_anom = int(0.20 * n_total_anom)
    
    val_anomaly = anomalies[:n_val_anom]
    test_anomaly = anomalies[n_val_anom:]
    
    train_set = train_normal
    val_set = np.vstack([val_normal, val_anomaly])
    test_set = np.vstack([test_normal, test_anomaly])
    
    val_set = val_set[np.random.permutation(len(val_set))]
    test_set = test_set[np.random.permutation(len(test_set))]
    
    return train_set, val_set, test_set

def compute_metrics(scores, y_true, thresh=None, pos_label=1):
    y_true = y_true.astype(int)
    results = {}
    
    try:
        results["roc_auc"] = sk_metrics.roc_auc_score(y_true, scores)
        results["avg_pr"] = sk_metrics.average_precision_score(y_true, scores)
    except ValueError:
        results["roc_auc"] = 0.0
        results["avg_pr"] = 0.0

    if thresh is not None:
        y_pred = (scores >= thresh).astype(int)
        
        accuracy = sk_metrics.accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = sk_metrics.precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0
        )
        cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        results.update({
            "acc": accuracy,
            "prec": precision,
            "rec": recall,
            "f1": f1,
            "cm": cm
        })
        
    return results

def estimate_optimal_threshold(val_scores, y_val, pos_label=1, nq=100):
    ratio = 100 * np.mean(y_val == pos_label)
    if ratio == 0: ratio = 50 
    
    q = np.linspace(max(0, ratio - 10), min(ratio + 10, 100), nq)
    thresholds = np.percentile(val_scores, q)

    best_f1 = -1
    best_thresh = thresholds[0]

    for thresh in thresholds:
        metrics = compute_metrics(val_scores, y_val, thresh, pos_label)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_thresh = thresh

    return best_thresh

def clustering_metrics(X, labels_pred, labels_true=None):
    results = {}
    
    if len(set(labels_pred)) > 1:
        results["Silhouette"] = silhouette_score(X, labels_pred)
        results["Davies-Bouldin"] = davies_bouldin_score(X, labels_pred)
        results["Calinski-Harabasz"] = calinski_harabasz_score(X, labels_pred)
    else:
        results["Silhouette"] = -1
        results["Davies-Bouldin"] = -1
        results["Calinski-Harabasz"] = -1

    if labels_true is not None:
        results["ARI"] = adjusted_rand_score(labels_true, labels_pred)
        results["NMI"] = normalized_mutual_info_score(labels_true, labels_pred)

    return results

def plot_loss(train_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_2d_embedding(X_2d, labels, title=""):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()

    

def clustering_metrics(DATA,LABELS_PREDICT, LABELS_TRUE) :
    silh_score = silhouette_score(DATA,LABELS_PREDICT,random_state=42) 

    db_score = davies_bouldin_score(DATA,labels=LABELS_PREDICT)
    
    ch_score = calinski_harabasz_score(DATA,labels=LABELS_PREDICT)
    
    ari_score = adjusted_rand_score(labels_true=LABELS_TRUE, labels_pred=LABELS_PREDICT)

    
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
    results_dict = {
        'labels_list': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
        'ari': [],
        'nmi': []
    }
    if algo_name.lower() == 'kmeans':
        for seed in range(n_seeds):
            model = algo_model_func(N_clusters=n_clusters, random_state=seed,)
            labels = model.fit_predict(X)
            results_dict['labels_list'].append(labels)
            
            results_dict['silhouette'].append(silhouette_score(X, labels))
            results_dict['davies_bouldin'].append(davies_bouldin_score(X, labels))
            results_dict['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
            
            if y_true is not None:
                results_dict['ari'].append(adjusted_rand_score(y_true, labels))
                results_dict['nmi'].append(normalized_mutual_info_score(y_true, labels))
    elif algo_name.lower() == 'dbscan':
        eps = algo_params.get('eps', 0.5)
        min_samples = algo_params.get('min_samples', 5)
        
        for seed in range(n_seeds):
            model = algo_model_func(MIN_SAMPLES=min_samples, EPS=eps)
            labels = model.fit_predict(X)
            results_dict['labels_list'].append(labels)
            
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            
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
    
    if true_labels is not None:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax = [ax]
    
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
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé : {save_path}")
    
    plt.show()