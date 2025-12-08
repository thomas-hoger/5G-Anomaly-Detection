import logging
import pickle

import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

logger = logging.getLogger(__name__)

class GNNTrainer:
    """Trainer pour BaselineAE avec MLPDecoder"""

    def __init__(self,  # noqa: PLR0913
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

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

        self.train_history = []
        self.val_history   = []
        self.lr_history    = []

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    # -----------------------------------------------------------
    # TRAINING STEP
    # -----------------------------------------------------------
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0

        train_bar = tqdm(self.train_loader, desc="Training", leave=False, ncols=100)
        batch_history = []

        for batch in train_bar:
            batch = batch.to(self.device)  # noqa: PLW2901
            self.optimizer.zero_grad()

            z = self.model(batch.x, batch.edge_index, batch.edge_attr)

            batch.latent_space = z
            batch_history.append(batch)

            loss = self.model.recon_loss(z, batch.edge_index)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / num_batches, batch_history

    # -----------------------------------------------------------
    # VALIDATION STEP
    # -----------------------------------------------------------
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        val_bar = tqdm(self.val_loader, desc="Validation", leave=False, ncols=100)
        batch_history = []

        with torch.no_grad():
            for batch in val_bar:
                batch = batch.to(self.device)  # noqa: PLW2901

                z = self.model(batch.x, batch.edge_index, batch.edge_attr)

                batch.latent_space = z
                batch_history.append(batch)

                loss = self.model.recon_loss(z, batch.edge_index)

                total_loss += loss.item()
                num_batches += 1
                val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        return total_loss / num_batches, batch_history

    # -----------------------------------------------------------
    # TRAIN LOOP
    # -----------------------------------------------------------
    def train(self,
              num_epochs: int = 100,
              early_stopping_patience: int = 20,
              save_path: str | None = None):

        logger.info(f"Début entraînement: {num_epochs} époques")
        logger.info(f"Device: {self.device}")

        epoch_bar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch", ncols=120)

        for epoch in epoch_bar:
            train_loss, train_batches_history = self.train_epoch()
            val_loss, val_batches_history     = self.validate_epoch()

            self.scheduler.step(val_loss)

            self.train_history.append(train_loss)
            self.val_history.append(val_loss)
            lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(lr)

            epoch_bar.set_postfix({
                'train': f"{train_loss:.4f}",
                'val': f"{val_loss:.4f}",
                'best': f"{self.best_val_loss:.4f}",
                'lr': f"{lr:.2e}"
            })

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if save_path:
                    save_file = f"{save_path}/checkpoints/model_{epoch}.pth"
                    torch.save(self.model.state_dict(), save_file)

                    with open(f"{save_path}/model_data_history/train/{epoch}.pkl", 'wb') as f:
                        pickle.dump(train_batches_history, f)
                    with open(f"{save_path}/model_data_history/val/{epoch}.pkl", 'wb') as f:
                        pickle.dump(val_batches_history, f)

            else:
                self.patience_counter += 1

            if self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping à l’époque {epoch}.")
                break

        logger.info("Entraînement terminé.")
        logger.info(f"Meilleure validation : {self.best_val_loss:.4f}")

        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_val_loss": self.best_val_loss
        }
