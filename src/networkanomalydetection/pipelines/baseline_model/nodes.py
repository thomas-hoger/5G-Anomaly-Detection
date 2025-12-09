"""
Nœud Kedro pour le pipeline d'entraînement GNN
"""
import logging

import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GAE
from torch_geometric.utils import negative_sampling

from networkanomalydetection.core.learning.models.baseline import (
    # BaselineAE,
    GINEEncoder,
    # SimpleDecoder,
)
from networkanomalydetection.core.learning.train import GNNTrainer

logger = logging.getLogger(__name__)


def train_gnn_model(
    train_loader: Data,
    val_loader: Data,
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
    node_dim = len(train_loader.x[0])
    edge_dim = len(train_loader.edge_attr[0])
    hidden_dim = 64
    out_dim = 32

    train_loader = DataLoader(train_loader, batch_size=training_params["batch_size"], shuffle=False)
    val_loader = DataLoader(val_loader, batch_size=training_params["batch_size"], shuffle=False)

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
    history, last_state_dict = trainer.train(
        num_epochs=training_params["num_epochs"],
        early_stopping_patience=training_params["early_stopping_patience"],
        save_path= "./data"
    )

    logger.info("Entraînement terminé avec succès")

    return {
        "training_history"   : history,
        "training_params"    : training_params
    }, last_state_dict

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

def test_gnn_model(test_loader: DataLoader, last_state_dict: dict):

    logger.info("Début du test du GNN")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device utilisé: {device}")

    # ------------------
    # PARAMÈTRES DU MODÈLE
    # ------------------
    node_dim = len(test_loader.x[0])
    edge_dim = len(test_loader.edge_attr[0])
    hidden_dim = 64
    out_dim = 32

    # ------------------
    # ENCODER / DECODER
    # ------------------
    encoder = GINEEncoder(node_dim, edge_dim, hidden_dim, out_dim)
    model = GAE(encoder).to(device)

    model.load_state_dict(last_state_dict)

    model.eval()
    total_loss = 0

    with torch.no_grad():

        cfm_by_type = {}

        cfm = {
            "TP" : 0,
            "FN" : 0,
            "FP" : 0,
            "TN" : 0
        }

        # Latent space vector
        z = model(test_loader.x, test_loader.edge_index, test_loader.edge_attr.float())

        positive_edges = test_loader.edge_index
        positive_reconstruction = model.decoder(z, positive_edges, sigmoid=True)

        negative_edges = negative_sampling(positive_edges, z.size(0))
        negative_reconstruction = model.decoder(z, negative_edges, sigmoid=True)

        for k in range(len(positive_edges.T)):

            reconstruction = int(positive_reconstruction[k]
)
            u,v = positive_edges.T[k]

            attack = test_loader.is_attack[u] or test_loader.is_attack[v]
            attack_type = test_loader.type[u] or test_loader.type[v]

            if attack_type not in cfm_by_type:
                cfm_by_type[attack_type] = cfm.copy()

            if attack and reconstruction :
                cfm_by_type[attack_type]["TP"] += 1
            elif attack and (not reconstruction) :
                cfm_by_type[attack_type]["FN"] += 1
            elif (not attack) and reconstruction:
                cfm_by_type[attack_type]["FP"] += 1
            else:
                cfm_by_type[attack_type]["TN"] += 1

        for k in range(len(negative_edges.T)):

            reconstruction = 1 - int(negative_reconstruction[k]
)
            u,v = negative_edges.T[k]

            attack = test_loader.is_attack[u] or test_loader.is_attack[v]
            attack_type = test_loader.type[u] or test_loader.type[v]

            if attack_type not in cfm_by_type:
                cfm_by_type[attack_type] = cfm.copy()

            if attack and reconstruction :
                cfm_by_type[attack_type]["TP"] += 1
            elif attack and (not reconstruction) :
                cfm_by_type[attack_type]["FN"] += 1
            elif (not attack) and reconstruction:
                cfm_by_type[attack_type]["FP"] += 1
            else:
                cfm_by_type[attack_type]["TN"] += 1

        # Reconstruction loss
        total_loss = model.recon_loss(z, test_loader.edge_index).item()

    metrics_by_type = {}
    for type,cfm in cfm_by_type.items() :
        metrics_by_type[type] = {
            "accuracy" : (cfm["TP"] + cfm["TN"]) / (cfm["TP"] + cfm["TN"] + cfm["FP"] + cfm["FN"]),
            "precision" : cfm["TP"] / (cfm["TP"] + cfm["FP"]),
            "recall" : cfm["TP"] / (cfm["TP"] + cfm["FN"]),
            "f1" : cfm["TP"] / (cfm["TP"] + 0.5*(cfm["FP"] + cfm["FN"]))
        }
        metrics_by_type[type].update(cfm)

    return {
        "total_loss" : total_loss,
        "metrics_by_type" : metrics_by_type
    }
