import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from abc import ABC
import numpy as np

class AETrainer:
    def __init__(self, model, train_loader, device, args):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.epochs = args.epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.save_path = f"{args.save_dir}/{model.name}.pt"

    def train(self):
        train_losses = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc="training", leave=True)
            counter = 1
            for batch in progress_bar:
                inputs = batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                avg_loss = running_loss / counter
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}", epoch=epoch)
                progress_bar.update()
                counter += 1
            train_losses.append(avg_loss)
        torch.save(self.model.state_dict(), self.save_path)
        print(f"model saved to {self.save_path}")
        return train_losses

    @torch.no_grad()
    def compute_reconstruction_errors(self, val_loader):
        self.model.eval()
        errors = []
        for batch in val_loader:
            inputs = batch.to(self.device)
            outputs = self.model(inputs)
            batch_errors = torch.mean((outputs - inputs) ** 2, dim=1)
            errors.extend(batch_errors.cpu().numpy())
        return np.array(errors)

class TrainerBaseShallow(ABC):
    def __init__(self, model, data):
        self.data = data
        self.model = model
        self.name = "shallow"

    def train(self):
        self.model.clf.fit(self.data)

    def score(self, sample):
        return self.model.clf.predict(sample)

    def test(self, sample):
        score = self.score(sample)
        y_pred = np.where(score == 1, 0, score)
        return np.where(y_pred == -1, 1, y_pred)

    def get_params(self) -> dict:
        return {
            **self.model.get_params()
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)

class MCDropoutAETrainer:
    def __init__(self, model, train_loader, device, args):
        self.model = model.to(device)
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from abc import ABC
import numpy as np

class AETrainer:
    def __init__(self, model, train_loader, device, args):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.epochs = args.epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.save_path = f"{args.save_dir}/{model.name}.pt"

    def train(self):
        train_losses = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc="training", leave=True)
            counter = 1
            for batch in progress_bar:
                inputs = batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                avg_loss = running_loss / counter
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}", epoch=epoch)
                progress_bar.update()
                counter += 1
            train_losses.append(avg_loss)
        torch.save(self.model.state_dict(), self.save_path)
        print(f"model saved to {self.save_path}")
        return train_losses

    @torch.no_grad()
    def compute_reconstruction_errors(self, val_loader):
        self.model.eval()
        errors = []
        for batch in val_loader:
            inputs = batch.to(self.device)
            outputs = self.model(inputs)
            batch_errors = torch.mean((outputs - inputs) ** 2, dim=1)
            errors.extend(batch_errors.cpu().numpy())
        return np.array(errors)

class TrainerBaseShallow(ABC):
    def __init__(self, model, data):
        self.data = data
        self.model = model
        self.name = "shallow"

    def train(self):
        self.model.clf.fit(self.data)

    def score(self, sample):
        return self.model.clf.predict(sample)

    def test(self, sample):
        score = self.score(sample)
        y_pred = np.where(score == 1, 0, score)
        return np.where(y_pred == -1, 1, y_pred)

    def get_params(self) -> dict:
        return {
            **self.model.get_params()
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)

class MCDropoutAETrainer:
    def __init__(self, model, train_loader, device, args):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.epochs = args.epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.save_path = f"{args.save_dir}/{model.name}.pt"
        self.n_samples = 10

    def train(self):
        train_losses = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc="training", leave=True)
            counter = 1
            for batch in progress_bar:
                inputs = batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                avg_loss = running_loss / counter
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}", epoch=epoch)
                progress_bar.update()
                counter += 1
            train_losses.append(avg_loss)
        torch.save(self.model.state_dict(), self.save_path)
        print(f"model saved to {self.save_path}")
        return train_losses

    @torch.no_grad()
    def compute_mc_reconstruction_errors(self, val_loader):
        self.model.train()
        errors_mean = []
        errors_std = []
        
        for batch in val_loader:
            inputs = batch.to(self.device)
            
            predictions = []
            for _ in range(self.n_samples):
                outputs = self.model(inputs)
                predictions.append(outputs)
            
            predictions = torch.stack(predictions)
            
            mean_pred = predictions.mean(dim=0)
            batch_errors_mean = torch.mean((mean_pred - inputs) ** 2, dim=1)
            errors_mean.extend(batch_errors_mean.cpu().numpy())
            
            batch_uncertainty = predictions.std(dim=0).mean(dim=1)
            errors_std.extend(batch_uncertainty.cpu().numpy())
        
        return np.array(errors_mean), np.array(errors_std)