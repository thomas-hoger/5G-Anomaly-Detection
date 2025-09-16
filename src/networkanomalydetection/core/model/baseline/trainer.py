"""
Trainer pour GNN Auto-encodeur avec architecture tri-domaine corrigée
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GNNTrainer:
    """Trainer pour GNN Auto-encodeur avec logging complet et barres de progression"""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 gamma: float = 1.0):  # Ajout du paramètre gamma pour edges

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.alpha = alpha  # Poids pour nœuds centraux
        self.beta = beta    # Poids pour nœuds paramètres
        self.gamma = gamma  # Poids pour edges

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        #  Tracking des métriques tri-domaine
        self.train_history = {
            'total_loss': [],
            'central_loss': [],    # Nœuds centraux
            'param_loss': [],      # Nœuds paramètres
            'edge_loss': [],       # Edges
            'learning_rate': []
        }

        self.val_history = {
            'total_loss': [],
            'central_loss': [],
            'param_loss': [],
            'edge_loss': []
        }

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self) -> dict[str, float]:
        """Entraînement d'une époque avec barre de progression"""
        self.model.train()

        epoch_losses = {
            'total_loss': 0,
            'central_loss': 0,
            'param_loss': 0,
            'edge_loss': 0
        }

        num_batches = 0

        train_bar = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
            unit="batch",
            ncols=100
        )

        for batch in train_bar:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch)
            #  Passer les 3 paramètres de pondération
            losses = self.model.compute_loss(outputs, batch, self.alpha, self.beta, self.gamma)

            # Backward pass
            losses['total_loss'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulation des métriques
            for key, value in losses.items():
                epoch_losses[key] += value.item()

            num_batches += 1

            # Mise à jour barre de progression
            train_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'central': f"{losses['central_loss'].item():.4f}",
                'param': f"{losses['param_loss'].item():.4f}",
                'edge': f"{losses['edge_loss'].item():.4f}"
            })

        # Moyenne des losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate_epoch(self) -> dict[str, float]:
        """Validation d'une époque avec barre de progression"""
        self.model.eval()

        epoch_losses = {
            'total_loss': 0,
            'central_loss': 0,
            'param_loss': 0,
            'edge_loss': 0
        }

        num_batches = 0

        val_bar = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
            unit="batch",
            ncols=100
        )

        with torch.no_grad():
            for batch in val_bar:
                batch = batch.to(self.device)

                outputs = self.model(batch)
                #  Passer les 3 paramètres de pondération
                losses = self.model.compute_loss(outputs, batch, self.alpha, self.beta, self.gamma)

                for key, value in losses.items():
                    epoch_losses[key] += value.item()

                num_batches += 1

                # Mise à jour barre de progression
                val_bar.set_postfix({
                    'val_loss': f"{losses['total_loss'].item():.4f}",
                    'central': f"{losses['central_loss'].item():.4f}",
                    'param': f"{losses['param_loss'].item():.4f}",
                    'edge': f"{losses['edge_loss'].item():.4f}"
                })

        # Moyenne des losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def train(self,
              num_epochs: int = 100,
              early_stopping_patience: int = 20,
              log_interval: int = 10,
              save_path: str|None = None) -> dict:
        """Entraînement complet avec early stopping et barres de progression"""

        logger.info(f"Début entraînement: {num_epochs} époques max")
        logger.info(f"Device: {self.device}")
        logger.info(f"Paramètres: α={self.alpha}, β={self.beta}, γ={self.gamma}")  # ✅ Afficher gamma
        logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

        epoch_bar = tqdm(
            range(num_epochs),
            desc="Training Progress",
            unit="epoch",
            ncols=120
        )

        for epoch in epoch_bar:
            # Entraînement
            train_losses = self.train_epoch()

            # Validation
            val_losses = self.validate_epoch()

            # Update scheduler
            self.scheduler.step(val_losses['total_loss'])

            # Sauvegarde métriques
            for key, value in train_losses.items():
                self.train_history[key].append(value)

            for key, value in val_losses.items():
                self.val_history[key].append(value)

            self.train_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Mise à jour barre principale avec métriques
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_bar.set_postfix({
                'train': f"{train_losses['total_loss']:.4f}",
                'val': f"{val_losses['total_loss']:.4f}",
                'best': f"{self.best_val_loss:.4f}",
                'patience': f"{self.patience_counter}/{early_stopping_patience}",
                'lr': f"{current_lr:.2e}"
            })

            #  Logging détaillé tri-domaine
            if epoch % log_interval == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_losses['total_loss']:.4f} "
                    f"(Central: {train_losses['central_loss']:.4f}, "
                    f"Param: {train_losses['param_loss']:.4f}, "
                    f"Edge: {train_losses['edge_loss']:.4f}) | "
                    f"Val Loss: {val_losses['total_loss']:.4f} "
                    f"(Central: {val_losses['central_loss']:.4f}, "
                    f"Param: {val_losses['param_loss']:.4f}, "
                    f"Edge: {val_losses['edge_loss']:.4f}) | "
                    f"LR: {current_lr:.2e}"
                )

            # Early stopping
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.patience_counter = 0

                if save_path:
                    self.save_checkpoint(save_path, epoch, val_losses)
            else:
                self.patience_counter += 1

            if self.patience_counter >= early_stopping_patience:
                epoch_bar.set_description("Early Stopping")
                logger.info(f"Early stopping à l'époque {epoch}")
                break

        epoch_bar.close()

        logger.info("Entraînement terminé")
        logger.info(f"Meilleure validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Époques totales: {epoch + 1}")

        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss,
            'total_epochs': epoch + 1
        }

    def save_checkpoint(self, save_path: str, epoch: int, val_losses: dict):
        """Sauvegarder checkpoint du modèle"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_losses': val_losses,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'hyperparameters': {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,  #  Sauvegarder gamma
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint sauvé: {save_path}")

    #  CORRECTION MAJEURE : evaluate_final_metrics
    def evaluate_final_metrics(self) -> dict:
        """Évaluation finale avec métriques détaillées tri-domaine"""
        logger.info("Début évaluation finale des métriques")

        self.model.eval()

        #  Séparer les erreurs par domaine
        all_central_errors = []
        all_param_errors = []
        all_edge_errors = []

        eval_bar = tqdm(
            self.val_loader,
            desc="Final Evaluation",
            unit="batch",
            ncols=100
        )

        with torch.no_grad():
            for batch in eval_bar:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                scores = self.model.compute_anomaly_scores(outputs, batch)

                #  Utiliser les bonnes clés tri-domaine
                if len(scores['central_anomaly_scores']) > 0:
                    all_central_errors.extend(scores['central_anomaly_scores'].cpu().numpy())

                if len(scores['param_anomaly_scores']) > 0:
                    all_param_errors.extend(scores['param_anomaly_scores'].cpu().numpy())

                all_edge_errors.extend(scores['edge_anomaly_scores'].cpu().numpy())

                # Mise à jour compteurs dans la barre
                eval_bar.set_postfix({
                    'central': len(all_central_errors),
                    'param': len(all_param_errors),
                    'edge': len(all_edge_errors)
                })

        eval_bar.close()

        #  Calcul des métriques tri-domaine
        metrics = {}

        # Métriques nœuds centraux
        if all_central_errors:
            central_errors_array = np.array(all_central_errors)
            metrics['central_metrics'] = {
                'count': len(all_central_errors),
                'mean_error': np.mean(central_errors_array),
                'std_error': np.std(central_errors_array),
                'min_error': np.min(central_errors_array),
                'max_error': np.max(central_errors_array),
                'percentile_95': np.percentile(central_errors_array, 95),
                'percentile_99': np.percentile(central_errors_array, 99)
            }
            logger.info(f"Central metrics - Count: {len(all_central_errors)}, "
                       f"Mean: {metrics['central_metrics']['mean_error']:.4f}, "
                       f"95%ile: {metrics['central_metrics']['percentile_95']:.4f}")

        # Métriques nœuds paramètres
        if all_param_errors:
            param_errors_array = np.array(all_param_errors)
            metrics['param_metrics'] = {
                'count': len(all_param_errors),
                'mean_error': np.mean(param_errors_array),
                'std_error': np.std(param_errors_array),
                'min_error': np.min(param_errors_array),
                'max_error': np.max(param_errors_array),
                'percentile_95': np.percentile(param_errors_array, 95),
                'percentile_99': np.percentile(param_errors_array, 99)
            }
            logger.info(f"Param metrics - Count: {len(all_param_errors)}, "
                       f"Mean: {metrics['param_metrics']['mean_error']:.4f}, "
                       f"95%ile: {metrics['param_metrics']['percentile_95']:.4f}")

        # Métriques edges
        if all_edge_errors:
            edge_errors_array = np.array(all_edge_errors)
            metrics['edge_metrics'] = {
                'count': len(all_edge_errors),
                'mean_error': np.mean(edge_errors_array),
                'std_error': np.std(edge_errors_array),
                'min_error': np.min(edge_errors_array),
                'max_error': np.max(edge_errors_array),
                'percentile_95': np.percentile(edge_errors_array, 95),
                'percentile_99': np.percentile(edge_errors_array, 99)
            }
            logger.info(f"Edge metrics - Count: {len(all_edge_errors)}, "
                       f"Mean: {metrics['edge_metrics']['mean_error']:.4f}, "
                       f"95%ile: {metrics['edge_metrics']['percentile_95']:.4f}")

        logger.info("Évaluation finale terminée")

        return metrics

    # CORRECTION MAJEURE : compute_thresholds_with_progress
    def compute_thresholds_with_progress(self, percentile: float = 95) -> dict:
        """Calculer les seuils d'anomalie tri-domaine"""
        logger.info(f"Calcul des seuils d'anomalie ({percentile}ème percentile)")

        self.model.eval()

        #  Séparer les erreurs par domaine
        all_central_errors = []
        all_param_errors = []
        all_edge_errors = []

        threshold_bar = tqdm(
            self.val_loader,
            desc="Computing Thresholds",
            unit="batch",
            ncols=100
        )

        with torch.no_grad():
            for batch in threshold_bar:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                scores = self.model.compute_anomaly_scores(outputs, batch)

                #  Utiliser les bonnes clés tri-domaine
                if len(scores['central_anomaly_scores']) > 0:
                    all_central_errors.extend(scores['central_anomaly_scores'].cpu().numpy())

                if len(scores['param_anomaly_scores']) > 0:
                    all_param_errors.extend(scores['param_anomaly_scores'].cpu().numpy())

                all_edge_errors.extend(scores['edge_anomaly_scores'].cpu().numpy())

                threshold_bar.set_postfix({
                    'samples': len(all_central_errors) + len(all_param_errors) + len(all_edge_errors)
                })

        threshold_bar.close()

        #  Calculer seuils tri-domaine
        thresholds = {}

        if all_central_errors:
            thresholds['central_threshold'] = np.percentile(all_central_errors, percentile)
            logger.info(f"Seuil nœuds centraux ({percentile}ème percentile): {thresholds['central_threshold']:.4f}")

        if all_param_errors:
            thresholds['param_threshold'] = np.percentile(all_param_errors, percentile)
            logger.info(f"Seuil nœuds paramètres ({percentile}ème percentile): {thresholds['param_threshold']:.4f}")

        if all_edge_errors:
            thresholds['edge_threshold'] = np.percentile(all_edge_errors, percentile)
            logger.info(f"Seuil edges ({percentile}ème percentile): {thresholds['edge_threshold']:.4f}")

        return thresholds

    #  CORRECTION : plot_training_curves pour tri-domaine
    def plot_training_curves(self, save_path: str|None = None):
        """Créer les courbes d'entraînement tri-domaine (2x2)"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(len(self.train_history['total_loss']))

        # Loss totale
        axes[0, 0].plot(epochs, self.train_history['total_loss'], label='Train', color='blue')
        axes[0, 0].plot(epochs, self.val_history['total_loss'], label='Validation', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Central loss
        axes[0, 1].plot(epochs, self.train_history['central_loss'], label='Train', color='blue')
        axes[0, 1].plot(epochs, self.val_history['central_loss'], label='Validation', color='red')
        axes[0, 1].set_title('Central Nodes Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Param loss
        axes[1, 0].plot(epochs, self.train_history['param_loss'], label='Train', color='green')
        axes[1, 0].plot(epochs, self.val_history['param_loss'], label='Validation', color='orange')
        axes[1, 0].set_title('Parameter Nodes Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate (on garde la même position)
        axes[1, 1].plot(epochs, self.train_history['learning_rate'], color='purple')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')

        plt.suptitle('GNN AutoEncoder - Training Curves (Tri-domaine)', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Courbes sauvées: {save_path}")

        plt.show()
