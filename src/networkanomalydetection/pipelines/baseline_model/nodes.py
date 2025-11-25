"""
Nœud Kedro pour le pipeline d'entraînement GNN
"""
import logging

import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE

from networkanomalydetection.core.learning.models.baseline import (
    # BaselineAE,
    GINEEncoder,
    # SimpleDecoder,
)
from networkanomalydetection.core.learning.train import GNNTrainer

logger = logging.getLogger(__name__)


def train_gnn_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_params: dict[str, any]
) -> dict[str, any]:

    logger.info("Début entraînement du modèle GNN")

    # ------------------
    # DEVICE
    # ------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device utilisé: {device}")

    # ------------------
    # PARAMÈTRES DU MODÈLE
    # ------------------
    node_dim = len(train_loader.dataset[0].x[0])
    edge_dim = len(train_loader.dataset[0].edge_attr[0])
    hidden_dim = 64
    out_dim = 32

    # ------------------
    # ENCODER / DECODER
    # ------------------
    encoder = GINEEncoder(node_dim, edge_dim, hidden_dim, out_dim)
    # decoder = SimpleDecoder(out_dim)
    # model   = BaselineAE(encoder, decoder).to(device)
    model = GAE(encoder).to(device)

    logger.info(
        f"Modèle initialisé: {sum(p.numel() for p in model.parameters())} paramètres"
    )

    # ------------------
    # TRAINER
    # ------------------
    trainer = GNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=training_params["learning_rate"],
        weight_decay=training_params["weight_decay"]
    )

    # ------------------
    # ENTRAÎNEMENT
    # ------------------
    history = trainer.train(
        num_epochs=training_params["num_epochs"],
        early_stopping_patience=training_params["early_stopping_patience"],
        save_path= "./data/checkpoints"
    )

    logger.info("Entraînement terminé avec succès")

    return {
        "training_history": history,
        "training_params": training_params,
    }

def plot_train(history:dict):

    plt.figure(figsize=(10, 5))
    plt.plot(history['training_history']['train_history'], label='Train Loss')
    plt.plot(history['training_history']['val_history'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('./data/report/figures/gnn_training_loss.png')
