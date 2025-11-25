import argparse
import numpy as np
from pathlib import Path
from trainers import AETrainer, TrainerBaseShallow, MCDropoutAETrainer
from models import AE, IF, MCDropoutAE
from utils import estimate_optimal_threshold, compute_metrics, plot_loss, split_ecg_data, compute_anomaly_metrics
from torch.utils.data import DataLoader
import torch

np.random.seed(42)

directory_model = "checkpoints/"
directory_output = "outputs/"

Path(directory_model).mkdir(parents=True, exist_ok=True)
Path(directory_output).mkdir(parents=True, exist_ok=True)


def train_autoencoder(args, data):
    print("\n" + "="*60)
    print("AUTO-ENCODEUR - Détection d'Anomalies")
    print("="*60)
    
    train_set, val_set, test_set = split_ecg_data(data)
    
    x_train = train_set[:, :-1]
    x_val = val_set[:, :-1]
    y_val = val_set[:, -1]
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    
    print(f"\nDonnées splittées")
    print(f"  - TRAIN: {x_train.shape} (normales seulement)")
    print(f"  - VAL: {x_val.shape}")
    print(f"  - TEST: {x_test.shape}")
    
    in_features = x_train.shape[1]
    
    print(f"\nPréparation des données")
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    
    train_loader = DataLoader(x_train_tensor, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(x_val_tensor, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(x_test_tensor, batch_size=args.batch_size, shuffle=False)
    
    print(f"\nCréation du modèle AE")
    ae_model = AE(in_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    print(f"\nEntraînement (epochs={args.epochs})")
    trainer = AETrainer(ae_model, train_loader, device, args)
    train_losses = trainer.train()
    
    print(f"\nCalcul des erreurs de reconstruction")
    val_scores = trainer.compute_reconstruction_errors(val_loader)
    test_scores = trainer.compute_reconstruction_errors(test_loader)
    
    print(f"\nRecherche du seuil optimal")
    thresh = estimate_optimal_threshold(val_scores, y_val, pos_label=1, nq=100)
    print(f"  Seuil optimal: {thresh:.4f}")
    
    print(f"\nMétriques sur TEST")
    metrics = compute_metrics(test_scores, y_test, thresh, pos_label=1)
    
    for metric_name, metric_value in metrics.items():
        if metric_name != "cm":
            print(f"  {metric_name}: {metric_value:.4f}")
    
    print(f"\nMatrice de confusion:\n{metrics['cm']}")
    
    print(f"\nVisualisation des losses d'entraînement")
    plot_loss(train_losses)
    
    return metrics, train_losses


def train_isolation_forest(args, data):
    print("\n" + "="*60)
    print("ISOLATION FOREST - Détection d'Anomalies")
    print("="*60)
    
    train_set, val_set, test_set = split_ecg_data
    (data)
    
    x_train = train_set[:, :-1]
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    
    print(f"\nDonnées splittées:")
    print(f"  - TRAIN: {x_train.shape} (normales seulement)")
    print(f"  - TEST: {x_test.shape}")
    print(f"    • {(y_test == 0).sum()} normales")
    print(f"    • {(y_test == 1).sum()} anomalies")
    
    print(f"\nEntraînement IF (n_estimators={args.n_estimators})...")
    if_model = IF(args)
    if_trainer = TrainerBaseShallow(if_model, x_train)
    if_trainer.train()
    print(f"Entraînement terminé")
    
    print(f"\nPrédictions sur TEST")
    preds = if_trainer.score(x_test)
    y_pred = np.where(preds == -1, 1, 0)
    print(f"Prédictions faites")
    
    print(f"\nMétriques:")
    metrics = compute_anomaly_metrics(y_pred, y_test, pos_label=1)
    
    for metric_name, metric_value in metrics.items():
        if metric_name != "cm":
            print(f"  {metric_name}: {metric_value:.4f}")
    
    print(f"\nMatrice de confusion:\n{metrics['cm']}")
    
    return metrics

def train_mc_dropout_autoencoder(args, data):
    print("\n" + "="*60)
    print("MC DROPOUT AUTO-ENCODEUR - Détection d'Anomalies")
    print("="*60)
    
    train_set, val_set, test_set = split_ecg_data(data)
    
    x_train = train_set[:, :-1]
    x_val = val_set[:, :-1]
    y_val = val_set[:, -1]
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    
    print(f"\nDonnées splittées:")
    print(f"  - TRAIN: {x_train.shape} (normales seulement)")
    print(f"  - VAL: {x_val.shape}")
    print(f"  - TEST: {x_test.shape}")
    
    in_features = x_train.shape[1]
    
    print(f"\nPréparation des données")
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    
    train_loader = DataLoader(x_train_tensor, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(x_val_tensor, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(x_test_tensor, batch_size=args.batch_size, shuffle=False)
    
    print(f"\nCréation du modèle MC Dropout AE")
    mc_model = MCDropoutAE(in_features, dropout_rate=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    print(f"\nEntraînement (epochs={args.epochs})           ")
    trainer = MCDropoutAETrainer(mc_model, train_loader, device, args)
    
    print(f"\nMétriques sur TEST")
    metrics = compute_metrics(test_scores, y_test, thresh, pos_label=1)
    
    for metric_name, metric_value in metrics.items():
        if metric_name != "cm":
            print(f"  {metric_name}: {metric_value:.4f}")
    
    print(f"\nMatrice de confusion:\n{metrics['cm']}")
    
    print(f"\nVisualisation des losses d'entraînement")
    plot_loss(train_losses)
    
    return metrics, train_losses

if __name__ == "__main__":
    ecg_path = r"C:\Users\jenja\Desktop\Session automne 2025\Science_données\TP2 - IFT 599\ecg.npz"
    ecg_data = np.load(ecg_path)
    data = ecg_data['ecg']
    
    print(f"Données chargées: {data.shape}")
    print(f"  - Label 0 (normales): {(data[:, -1] == 0).sum()}")
    print(f"  - Label 1 (anomalies): {(data[:, -1] == 1).sum()}")
    
    parser = argparse.ArgumentParser(description="Anomaly Detection", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00001)
    parser.add_argument('-w', '--weight_decay', type=float, default=0.001)
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-c', '--n_components', type=int, default=3)
    parser.add_argument('-es', '--n_estimators', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    metrics_if = train_isolation_forest(args, data)
    metrics_ae, losses_ae = train_autoencoder(args, data)
    metrics_mc, losses_mc = train_mc_dropout_autoencoder(args, data)
    
    print("\n" + "="*60)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*60)
    print("\nIsolation Forest:")
    for k, v in metrics_if.items():
        if k != "cm":
            print(f"  {k}: {v:.4f}")
    print("\nAuto-encodeur:")
    for k, v in metrics_ae.items():
        if k != "cm":
            print(f"  {k}: {v:.4f}")
    print("\nMC Dropout Auto-encodeur:")
    for k, v in metrics_mc.items():
        if k != "cm":
            print(f"  {k}: {v:.4f}")