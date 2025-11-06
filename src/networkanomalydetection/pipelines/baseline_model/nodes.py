"""
Nœuds Kedro pour le pipeline d'entraînement GNN
"""
import logging

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE

from networkanomalydetection.core.learning.models.baseline import GCNEncoder
from networkanomalydetection.core.learning.train import GNNTrainer

logger = logging.getLogger(__name__)

def train_gnn_model(train_loader: DataLoader, val_loader: DataLoader, model_params: dict[str, any], training_params: dict[str, any]) -> dict[str, any]:

    logger.info("Début entraînement du modèle GNN")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device utilisé: {device}")

    # Initialiser le modèle
    model = GAE(GCNEncoder(num_features, out_channels))
    logger.info(f"Modèle initialisé: {sum(p.numel() for p in model.parameters())} paramètres")

    # Initialiser le trainer
    trainer = GNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=training_params['learning_rate'],
        weight_decay=training_params['weight_decay'],
        alpha=training_params['alpha'],
        beta=training_params['beta']
    )

    # Entraînement
    history = trainer.train(
        num_epochs=training_params['num_epochs'],
        early_stopping_patience=training_params['early_stopping_patience'],
        log_interval=10,
        save_path="./test_poids"
    )

    logger.info("Entraînement terminé avec succès")

    trainer.training_curves("./test_curve_train.jpg")

    return {
        'model': model,
        'trainer': trainer,
        'training_history': history,
        'model_params': model_params,
        'training_params': training_params,
    }

