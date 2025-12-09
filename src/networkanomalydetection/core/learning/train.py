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
        self.optimizer.zero_grad()

        z = self.model(self.train_loader.x, self.train_loader.edge_index, self.train_loader.edge_attr.float())
        loss = self.model.recon_loss(z, self.train_loader.edge_index)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), z

    # -----------------------------------------------------------
    # VALIDATION STEP
    # -----------------------------------------------------------
    def validate_epoch(self):

        self.model.eval()
        with torch.no_grad():
            z = self.model(self.val_loader.x, self.val_loader.edge_index, self.val_loader.edge_attr)
            loss = self.model.recon_loss(z, self.val_loader.edge_index)

        return loss.item(), z

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

            # train
            train_loss, z_train = self.train_epoch()
            self.train_history.append(train_loss)

            # val
            val_loss, z_val = self.validate_epoch()
            self.val_history.append(val_loss)

            self.scheduler.step(val_loss)
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
                        pickle.dump(z_train, f)
                    with open(f"{save_path}/model_data_history/val/{epoch}.pkl", 'wb') as f:
                        pickle.dump(z_val, f)

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
        }, self.model.state_dict()
