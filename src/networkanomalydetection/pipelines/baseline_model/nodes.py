"""
Nœud Kedro pour le pipeline d'entraînement GNN
"""
import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patheffects
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
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

    train_loader = NeighborLoader(train_loader, [30, 30], train_loader.is_central, batch_size=training_params["batch_size"],shuffle=False)
    val_loader = NeighborLoader(val_loader, [30, 30], val_loader.is_central, batch_size=training_params["batch_size"],shuffle=False)

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

# def plot_pca_latent_space(latent_space_history:dict[str,list[dict]], histo_results:dict):

#     for file, loader in latent_space_history.items():

#         first_batch = next(iter(loader()))

#         # compute mean cosine similarity
#         S = cosine_similarity(first_batch.detach())
#         mask = ~np.eye(S.shape[0], dtype=bool)
#         mean_similarity = S[mask].mean()

#         # projection 2D avec t-SNE
#         tsne = TSNE(
#             n_components=2,
#             perplexity=30,
#             learning_rate='auto',
#             init='random',
#             max_iter=1000,
#             verbose=1
#         )

#         X_2d = tsne.fit_transform(first_batch.detach())

#         labels = []
#         for label_list in data_batch.label:
#             labels += label_list

#         # clustering
#         k = 10
#         kmeans = KMeans(n_clusters=k).fit(X_2d)
#         cluster_ids = kmeans.labels_

#         # colormap
#         cmap = plt.get_cmap("tab20")
#         palette = [cmap(i/(k-1)) for i in range(k)]  # normalisation [0,1]
#         plt.figure(figsize=(12, 12))

#         # scatter
#         point_colors = [palette[c] for c in cluster_ids]
#         plt.scatter(X_2d[:,0], X_2d[:,1], c=point_colors, s=3)

#         # nombre de labels par cluster
#         n_labels = 6
#         for c in range(k):
#             pts = np.where(cluster_ids==c)[0]
#             if len(pts)==0:
#                 continue

#             # sélection jusqu'à n_labels avec label non nul
#             selected = []
#             for idx in pts:
#                 lbl = labels[idx]
#                 if lbl not in (None, "", [], 0):
#                     selected.append(idx)
#                 if len(selected) >= n_labels:
#                     break
#             if len(selected)==0:
#                 continue

#             cluster_color = palette[c]

#             # annotation
#             for i, idx in enumerate(selected):
#                 lbl = str(labels[idx])
#                 x, y = X_2d[idx]
#                 offset_y = i*2  # léger décalage pour ne pas superposer

#                 txt = plt.text(
#                     x, y+offset_y, lbl,
#                     fontsize=5,
#                     color=cluster_color,
#                     weight="bold",
#                     ha="center",
#                     va="center"
#                 )

#                 # bordure pour lisibilité
#                 txt.set_path_effects([
#                     patheffects.Stroke(linewidth=3, foreground="white"),
#                     patheffects.Normal()
#                 ])

#         plt.axis("off")
#         plt.title(f'Latent Space Visualization (t-SNE) - Mean Cosine Similarity: {mean_similarity:.4f}')
#         plt.savefig(f'./data/report/figures/gnn_latent_space_{file}.png')

def test_gnn_model(test_loader: Data, last_state_dict: dict):

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

    batch_size=32

    test_loader = NeighborLoader(test_loader, [30, 30], test_loader.is_central, batch_size=batch_size, shuffle=False)

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

        test_z_history = []
        for batch in test_loader:

            batch = batch.to(device)  # noqa: PLW2901

            # Latent space vector
            z = model(batch.x, batch.edge_index, batch.edge_attr)
            test_z_history.append(z)

            positive_edges = batch.edge_index
            positive_reconstruction = model.decoder(z, positive_edges, sigmoid=True)

            negative_edges = negative_sampling(positive_edges, z.size(0))
            negative_reconstruction = model.decoder(z, negative_edges, sigmoid=True)

            for k in range(len(positive_edges.T)):

                reconstruction = int(positive_reconstruction[k])
                u,v = positive_edges.T[k]

                attack = batch.is_attack[u].item() > 0 or batch.is_attack[v].item() > 0
                attack_type = batch.type[u].item() or batch.type[v].item()

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

                reconstruction = 1 - int(negative_reconstruction[k])
                u,v = negative_edges.T[k]

                attack = batch.is_attack[u].item() > 0 or batch.is_attack[v].item() > 0
                attack_type = batch.type[u].item() or batch.type[v].item()

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
            total_loss = model.recon_loss(z, batch.edge_index).item()

    metrics_by_type = {}
    for type,cfm in cfm_by_type.items() :
        metrics_by_type[type] = {
            "accuracy" : (cfm["TP"] + cfm["TN"]) / (cfm["TP"] + cfm["TN"] + cfm["FP"] + cfm["FN"]),
            "precision" : cfm["TP"] / (cfm["TP"] + cfm["FP"]),
            "recall" : cfm["TP"] / (cfm["TP"] + cfm["FN"]),
            "f1" : cfm["TP"] / (cfm["TP"] + 0.5*(cfm["FP"] + cfm["FN"]))
        }
        metrics_by_type[type].update(cfm)

    with open("./data/model_data_history/test/loader.pkl", 'wb') as f:
        pickle.dump(test_loader, f)

    with open("./data/model_data_history/test/test.pkl", 'wb') as f:
        pickle.dump(test_z_history, f)

    return {
        "total_loss" : total_loss,
        "metrics_by_type" : metrics_by_type
    }