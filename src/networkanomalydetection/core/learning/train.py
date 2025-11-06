"""
Trainer pour GNN Auto-encodeur avec architecture tri-domaine corrigée
"""
import logging

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

class GNNTrainer:
    """Trainer pour GNN Auto-encodeur avec logging complet et barres de progression"""

    def __init__(self,  # noqa: PLR0913
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
        self.train_history = []
        self.val_history = []
        self.lr_history = []

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self) -> dict[str, float]:
        """Entraînement d'une époque avec barre de progression"""
        self.model.train()

        num_batches = 0
        tot_loss    = 0

        train_bar = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
            unit="batch",
            ncols=100
        )

        for batch in train_bar:

            batch_device = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch_device)

            #  Passer les 3 paramètres de pondération
            z = self.model(batch_device)
            loss = self.model.recon_loss(z, train_pos_edge_index)
    
            loss = self.model.compute_loss(outputs, batch_device, self.alpha, self.beta, self.gamma)
            tot_loss += loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            num_batches += 1

            # Mise à jour barre de progression
            train_bar.set_postfix({
                'loss': f"{loss:.4f}",
            })

        return tot_loss / num_batches

    def validate_epoch(self) -> dict[str, float]:
        """Validation d'une époque avec barre de progression"""
        self.model.eval()

        tot_loss    = 0
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
                batch_device = batch.to(self.device)

                outputs = self.model(batch_device)
                #  Passer les 3 paramètres de pondération
                loss = self.model.compute_loss(outputs, batch_device, self.alpha, self.beta, self.gamma)
                tot_loss += loss

                num_batches += 1

                # Mise à jour barre de progression
                val_bar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                })

        return tot_loss / num_batches

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
            train_loss = self.train_epoch()

            # Validation
            val_loss = self.validate_epoch()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Sauvegarde métriques
            self.train_history.append(train_loss)
            self.val_history.append(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)

            # Mise à jour barre principale avec métriques

            epoch_bar.set_postfix({
                'train': f"{train_loss:.4f}",
                'val': f"{val_loss:.4f}",
                'best': f"{self.best_val_loss:.4f}",
                'patience': f"{self.patience_counter}/{early_stopping_patience}",
                'lr': f"{current_lr:.2e}"
            })

            #  Logging détaillé tri-domaine
            if epoch % log_interval == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f} "
                    f"Val Loss: {val_loss:.4f} "
                    f"LR: {current_lr:.2e}"
                )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                if save_path:
                    self.save_checkpoint(save_path, epoch, val_loss)
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
            'lr_history' : self.lr_history,
            'hyperparameters': {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,  #  Sauvegarder gamma
            }
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint sauvé: {save_path}")

    def training_curves(self, save_path: str|None = None):
        """Créer les courbes d'entraînement tri-domaine (2x2)"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(len(self.train_history))

        # Loss totale
        axes[0, 0].plot(epochs, self.train_history, label='Train', color='blue')
        axes[0, 0].plot(epochs, self.val_history, label='Validation', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Learning rate (on garde la même position)
        axes[1, 1].plot(epochs, self.lr_history, color='purple')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')

        plt.suptitle('Training Curves', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Courbes sauvées: {save_path}")

        # plt.show()
