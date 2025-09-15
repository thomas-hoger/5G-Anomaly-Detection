"""
Nœuds Kedro pour le pipeline d'entraînement GNN
"""
import logging
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch

from networkanomalydetection.core.gnn.models.data_utils import (
    analyze_data_statistics,
    compute_anomaly_thresholds,
    create_data_loaders,
    temporal_train_val_split,
)
from networkanomalydetection.core.gnn.models.gnn_autoencoder import (
    EnhancedGNNAutoEncoder,
)
from networkanomalydetection.core.gnn.models.trainer import GNNTrainer

logger = logging.getLogger(__name__)


def prepare_training_split(training_data_dict: Dict[str, Any],
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.3) -> Dict[str, List]:
    """
    Séparer les données en train/validation avec split temporel
    
    Args:
        training_data_dict: Dict avec clé 'batches' contenant les données
        train_ratio: Proportion d'entraînement
        val_ratio: Proportion de validation
    
    Returns:
        Dict avec train_data et val_data
    """
    logger.info("Début préparation du split train/validation")

    # CORRECTION: Extraire la liste depuis le dictionnaire
    training_data = training_data_dict['batches']
    logger.info(f"Données extraites: {len(training_data)} batches")

    # Analyser les statistiques des données
    stats = analyze_data_statistics(training_data)

    # Split temporel
    train_data, val_data = temporal_train_val_split(
        training_data,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    logger.info(f"Split terminé: {len(train_data)} train, {len(val_data)} validation")

    return {
        'train_data': train_data,
        'val_data': val_data,
        'data_statistics': stats
    }


def train_gnn_model(train_val_split: Dict[str, List],
                   model_params: Dict[str, Any],
                   training_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entraîner le modèle GNN
    
    Args:
        train_val_split: Données splitées
        model_params: Paramètres du modèle
        training_params: Paramètres d'entraînement
    
    Returns:
        Dict avec modèle entraîné et métriques
    """
    logger.info("Début entraînement du modèle GNN")

    train_data = train_val_split['train_data']
    val_data = train_val_split['val_data']

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device utilisé: {device}")

    # Créer DataLoaders
    train_loader, val_loader = create_data_loaders(
        train_data,
        val_data,
        batch_size=training_params['batch_size'],
        num_workers=4
    )

    # Initialiser le modèle
    model = EnhancedGNNAutoEncoder(**model_params)
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
        log_interval=10
    )

    # Métriques finales
    final_metrics = trainer.evaluate_final_metrics()

    # Calcul des seuils d'anomalie
    thresholds = compute_anomaly_thresholds(model, val_loader, device)

    logger.info("Entraînement terminé avec succès")

    return {
        'model': model,
        'trainer': trainer,
        'training_history': history,
        'final_metrics': final_metrics,
        'anomaly_thresholds': thresholds,
        'model_params': model_params,
        'training_params': training_params,
        'data_statistics': train_val_split['data_statistics']
    }


def generate_training_plots(training_results: Dict[str, Any]) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Générer les graphiques d'entraînement
    
    Args:
        training_results: Résultats d'entraînement
    
    Returns:
        Tuple with three matplotlib figures: (training_curves, error_histograms, training_summary)
    """
    logger.info("Génération des graphiques d'entraînement")

    trainer = training_results['trainer']

    # Courbes d'entraînement
    training_curves_fig = create_training_curves(trainer)

    # Histogrammes des erreurs
    error_hist_fig = create_error_histograms(training_results)

    # Résumé d'entraînement
    summary_fig = create_training_summary(training_results)

    logger.info("Graphiques générés avec succès")

    # Return as tuple instead of dictionary
    return training_curves_fig, error_hist_fig, summary_fig

def create_training_curves(trainer):
    """Créer les courbes d'entraînement avec 4 graphiques (total, central, param, edge)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(len(trainer.train_history['total_loss']))

    # Loss totale
    axes[0, 0].plot(epochs, trainer.train_history['total_loss'], label='Train', color='blue', linewidth=2)
    axes[0, 0].plot(epochs, trainer.val_history['total_loss'], label='Validation', color='red', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Central loss (ex-Node loss)
    axes[0, 1].plot(epochs, trainer.train_history['central_loss'], label='Train', color='blue', linewidth=2)
    axes[0, 1].plot(epochs, trainer.val_history['central_loss'], label='Validation', color='red', linewidth=2)
    axes[0, 1].set_title('Central Nodes Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Param loss (nouveau!)
    axes[1, 0].plot(epochs, trainer.train_history['param_loss'], label='Train', color='green', linewidth=2)
    axes[1, 0].plot(epochs, trainer.val_history['param_loss'], label='Validation', color='orange', linewidth=2)
    axes[1, 0].set_title('Parameter Nodes Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Edge loss
    axes[1, 1].plot(epochs, trainer.train_history['edge_loss'], label='Train', color='purple', linewidth=2)
    axes[1, 1].plot(epochs, trainer.val_history['edge_loss'], label='Validation', color='brown', linewidth=2)
    axes[1, 1].set_title('Edge Loss', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('GNN AutoEncoder - Training Curves (Central + Param + Edge)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


# Mise à jour create_error_histograms pour 3 types d'erreurs
def create_error_histograms(training_results):
    """Créer histogrammes des erreurs - Central + Param + Edge"""
    model = training_results['model']
    trainer = training_results['trainer']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    all_central_errors = []
    all_param_errors = []
    all_edge_errors = []

    with torch.no_grad():
        for batch in trainer.val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            scores = model.compute_anomaly_scores(outputs, batch)

            # Central errors
            if len(scores['central_anomaly_scores']) > 0:
                all_central_errors.extend(scores['central_anomaly_scores'].cpu().numpy())

            # Param errors
            if len(scores['param_anomaly_scores']) > 0:
                all_param_errors.extend(scores['param_anomaly_scores'].cpu().numpy())

            # Edge errors
            all_edge_errors.extend(scores['edge_anomaly_scores'].cpu().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Histogramme erreurs centraux
    if all_central_errors:
        axes[0].hist(all_central_errors, bins=50, alpha=0.7, color='blue', edgecolor='navy')
        if 'central_threshold' in training_results['anomaly_thresholds']:
            axes[0].axvline(training_results['anomaly_thresholds']['central_threshold'],
                           color='red', linestyle='--', label='Seuil 95%', linewidth=2)
        axes[0].set_title('Distribution Erreurs Nœuds Centraux', fontweight='bold')
        axes[0].set_xlabel('Erreur de Reconstruction')
        axes[0].set_ylabel('Fréquence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Histogramme erreurs paramètres
    if all_param_errors:
        axes[1].hist(all_param_errors, bins=50, alpha=0.7, color='green', edgecolor='darkgreen')
        if 'param_threshold' in training_results['anomaly_thresholds']:
            axes[1].axvline(training_results['anomaly_thresholds']['param_threshold'],
                           color='red', linestyle='--', label='Seuil 95%', linewidth=2)
        axes[1].set_title('Distribution Erreurs Nœuds Paramètres', fontweight='bold')
        axes[1].set_xlabel('Erreur de Reconstruction')
        axes[1].set_ylabel('Fréquence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Histogramme erreurs edges
    if all_edge_errors:
        axes[2].hist(all_edge_errors, bins=50, alpha=0.7, color='purple', edgecolor='indigo')
        if 'edge_threshold' in training_results['anomaly_thresholds']:
            axes[2].axvline(training_results['anomaly_thresholds']['edge_threshold'],
                           color='red', linestyle='--', label='Seuil 95%', linewidth=2)
        axes[2].set_title('Distribution Erreurs Edges', fontweight='bold')
        axes[2].set_xlabel('Erreur de Reconstruction')
        axes[2].set_ylabel('Fréquence')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.suptitle('Distributions des Erreurs de Reconstruction (Tri-domaine)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def save_trained_model(training_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Préparer les données pour sauvegarde du modèle
    
    Args:
        training_results: Résultats d'entraînement
    
    Returns:
        Dict avec modèle et métadonnées
    """
    logger.info("Préparation sauvegarde du modèle entraîné")

    # Préparer checkpoint modèle
    model_checkpoint = {
        'model_state_dict': training_results['model'].state_dict(),
        'model_params': training_results['model_params'],
        'training_params': training_results['training_params'],
        'best_val_loss': training_results['training_history']['best_val_loss'],
        'anomaly_thresholds': training_results['anomaly_thresholds']
    }

    # Préparer métadonnées
    metadata = {
        'final_metrics': training_results['final_metrics'],
        'anoamaly_thresholds': training_results['anomaly_thresholds'],
        'training_summary': {
            'total_epochs': training_results['training_history']['total_epochs'],
            'best_val_loss': training_results['training_history']['best_val_loss'],
            'final_trin_loss': training_results['training_history']['train_history']['total_loss'][-1],
            'final_val_loss': training_results['training_history']['val_history']['total_loss'][-1]
        },
        'model_config': training_results['model_params'],
        'training_config': training_results['training_params'],
        'data_statistics': training_results['data_statistics']
    }

    logger.info("Données de sauvegarde préparées")

    return {
        'model_checkpoint': model_checkpoint,
        'metadata': metadata
    }

