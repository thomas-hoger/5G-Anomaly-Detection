"""
Nœuds Kedro pour pipeline GNN pré-entraînement

"""

import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, List, Tuple, Any
import logging
import numpy as np
from datetime import datetime
import time

from networkanomalydetection.core.gnn_pretraining.model import MultiDomainPretrainingGNN
from networkanomalydetection.core.gnn_pretraining.data_utils import (
    prepare_graph_data,
    create_training_batches,
    validate_graph_data
)

logger = logging.getLogger(__name__)

def validate_data_node(vectorized_graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """
    Valide les données pour pré-entraînement
    
    Args:
        vectorized_graph: Graphe vectorisé depuis le pipeline précédent
        
    Returns:
        Rapport de validation
    """
    
    logger.info(" Validation des données de pré-entraînement")
    
    if vectorized_graph is None:
        raise ValueError("Aucun graphe fourni pour validation")
    
    validation_results = validate_graph_data(vectorized_graph)
    
    if not validation_results['valid']:
        logger.error(" Validation échouée:")
        for issue in validation_results['issues'][:5]:
            logger.error(f"  - {issue}")
        raise ValueError(f"Validation échouée avec {len(validation_results['issues'])} problèmes")
    
    stats = validation_results['stats']
    logger.info(f" Validation réussie:")
    logger.info(f"   {stats['total_nodes']:,} nœuds")
    logger.info(f"   {stats['total_edges']:,} arêtes")
    logger.info(f"   {stats['unique_packets']} paquets uniques")
    logger.info(f"   Types de nœuds: {stats['node_types']}")
    
    validation_report = {
        'validation_status': 'PASSED',
        'timestamp': datetime.now().isoformat(),
        'data_stats': stats,
        'issues': validation_results['issues']
    }
    
    return validation_report

def prepare_data_node(vectorized_graph: nx.MultiDiGraph,
                     validation_report: Dict[str, Any],
                     parameters: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Prépare les données pour entraînement
    
    Args:
        vectorized_graph: Graphe vectorisé
        validation_report: Rapport de validation
        parameters: Paramètres depuis parameters.yml
        
    Returns:
        Tuple[prepared_data, preparation_report]
    """
    
    config = {
        'max_nodes_for_training': 50000,    # Limite pour éviter surcharge mémoire
        'batch_size': 1000,                 # Nœuds par sous-batch
        'validation_split': 0.2
    }
    
    if parameters:
        config.update(parameters)
    
    logger.info(f" Préparation des données avec config: {config}")
    
    if validation_report.get('validation_status') != 'PASSED':
        raise ValueError("Impossible de préparer: validation non réussie")
    
    # Préparer le graphe
    batch_data = prepare_graph_data(
        vectorized_graph, 
        max_nodes=config['max_nodes_for_training']
    )
    
    # Créer sous-batches
    training_batches = create_training_batches(
        batch_data,
        batch_size=config['batch_size']
    )
    
    # Split train/validation
    split_idx = int(len(training_batches) * (1 - config['validation_split']))
    train_batches = training_batches[:split_idx] if split_idx > 0 else training_batches
    val_batches = training_batches[split_idx:] if split_idx < len(training_batches) else training_batches[:1]
    
    prepared_data = {
        'train_batches': train_batches,
        'val_batches': val_batches
    }
    
    preparation_report = {
        'preparation_status': 'SUCCESS',
        'timestamp': datetime.now().isoformat(),
        'data_splits': {
            'total_batches': len(training_batches),
            'train_batches': len(train_batches),
            'val_batches': len(val_batches)
        },
        'config_used': config
    }
    
    logger.info(f" Préparation terminée: {len(train_batches)} batches train, {len(val_batches)} batches val")
    
    return prepared_data, preparation_report

def initialize_model_node(preparation_report: Dict[str, Any],
                         parameters: Dict[str, Any] = None) -> Tuple[MultiDomainPretrainingGNN, Dict[str, Any]]:
    """
    Initialise le modèle GNN
    
    Args:
        preparation_report: Rapport de préparation
        parameters: Paramètres modèle
        
    Returns:
        Tuple[model, initialization_report]
    """
    
    config = {
        'node_dim': 64,
        'edge_dim': 64,
        'hidden_dim': 128,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'random_seed': 42
    }
    
    if parameters:
        config.update(parameters)
    
    logger.info(f" Initialisation du modèle avec config: {config}")
    
    # Seed pour reproductibilité
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Créer le modèle
    model = MultiDomainPretrainingGNN(
        node_dim=config['node_dim'],
        edge_dim=config['edge_dim'],
        hidden_dim=config['hidden_dim']
    )
    
    # Déplacer sur device
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Compter paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f" Architecture modèle:")
    logger.info(f"   Paramètres totaux: {total_params:,}")
    logger.info(f"   Paramètres entraînables: {trainable_params:,}")
    logger.info(f"   Device: {device}")
    
    # Test dummy
    logger.info("🧪 Test du modèle...")
    try:
        dummy_batch = _create_dummy_batch(config)
        dummy_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in dummy_batch.items()}
        
        with torch.no_grad():
            output = model(dummy_batch)
            logger.info(" Test forward pass réussi")
    except Exception as e:
        logger.error(f" Test modèle échoué: {e}")
        raise
    
    initialization_report = {
        'initialization_status': 'SUCCESS',
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(device)
        },
        'config_used': config
    }
    
    logger.info(" Initialisation modèle terminée")
    
    return model, initialization_report

def train_model_node(model: MultiDomainPretrainingGNN,
                    prepared_data: Dict[str, Any],
                    initialization_report: Dict[str, Any],
                    parameters: Dict[str, Any] = None) -> Tuple[MultiDomainPretrainingGNN, Dict[str, Any]]:
    """
    Entraîne le modèle GNN
    
    Args:
        model: Modèle initialisé
        prepared_data: Données préparées
        initialization_report: Rapport d'initialisation
        parameters: Paramètres d'entraînement
        
    Returns:
        Tuple[trained_model, training_report]
    """
    
    config = {
        'learning_rate': 0.001,
        'num_epochs': 50,
        'patience': 10,
        'min_delta': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'log_interval': 5
    }
    
    if parameters:
        config.update(parameters)
    
    logger.info(f" Début pré-entraînement avec config: {config}")
    
    train_batches = prepared_data['train_batches']
    val_batches = prepared_data['val_batches']
    
    if not train_batches:
        raise ValueError("Aucun batch d'entraînement")
    
    logger.info(f" Setup: {len(train_batches)} batches train, {len(val_batches)} batches val")
    
    # Configuration entraînement
    device = torch.device(config['device'])
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.7)
    
    # Variables early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Historique
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': []
    }
    
    logger.info(" Début boucle d'entraînement...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        
        # Phase entraînement
        model.train()
        train_loss_acc = []
        
        for batch_idx, batch in enumerate(train_batches):
            
            batch = _move_batch_to_device(batch, device)
            
            optimizer.zero_grad()
            outputs = model(batch)
            losses = model.compute_pretraining_loss(outputs)
            total_loss = losses['total_loss']
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"Loss invalide batch {batch_idx}, skip")
                continue
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_acc.append(total_loss.item())
        
        avg_train_loss = np.mean(train_loss_acc) if train_loss_acc else float('inf')
        
        # Phase validation
        model.eval()
        val_loss_acc = []
        
        with torch.no_grad():
            for batch in val_batches:
                batch = _move_batch_to_device(batch, device)
                outputs = model(batch)
                losses = model.compute_pretraining_loss(outputs)
                
                val_loss = losses['total_loss'].item()
                if not (torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss))):
                    val_loss_acc.append(val_loss)
        
        avg_val_loss = np.mean(val_loss_acc) if val_loss_acc else avg_train_loss
        
        # Scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Historique
        training_history['train_losses'].append(avg_train_loss)
        training_history['val_losses'].append(avg_val_loss)
        training_history['learning_rates'].append(current_lr)
        
        # Logging
        if epoch % config['log_interval'] == 0:
            logger.info(f"Epoch {epoch+1}/{config['num_epochs']}:")
            logger.info(f"   Train Loss: {avg_train_loss:.6f}")
            logger.info(f"   Val Loss: {avg_val_loss:.6f}")
            logger.info(f"   LR: {current_lr:.2e}")
        
        # Early stopping
        if avg_val_loss < best_val_loss - config['min_delta']:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            logger.info(f" Early stopping après {epoch+1} époques")
            break
    
    # Charger meilleur modèle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(" Meilleur modèle chargé")
    
    total_time = time.time() - start_time
    
    # Rapport
    training_report = {
        'training_status': 'COMPLETED',
        'timestamp': datetime.now().isoformat(),
        'training_summary': {
            'total_epochs': len(training_history['train_losses']),
            'total_time_seconds': round(total_time, 2),
            'final_train_loss': round(training_history['train_losses'][-1], 6),
            'final_val_loss': round(training_history['val_losses'][-1], 6),
            'best_val_loss': round(best_val_loss, 6)
        },
        'training_history': training_history,
        'config_used': config
    }
    
    logger.info(" Pré-entraînement terminé!")
    logger.info(f"   Temps total: {total_time/60:.1f} minutes")
    logger.info(f"   Meilleure val loss: {best_val_loss:.6f}")
    
    return model, training_report

def save_model_node(model: MultiDomainPretrainingGNN,
                   training_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sauvegarde le modèle pré-entraîné
    
    Args:
        model: Modèle entraîné
        training_report: Rapport d'entraînement
        
    Returns:
        Rapport de sauvegarde
    """
    
    logger.info(" Sauvegarde du modèle...")
    
    # Nettoyer mémoire dynamique
    model.clear_memory()
    
    save_report = {
        'save_status': 'SUCCESS',
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'training_epochs': training_report['training_summary']['total_epochs'],
            'final_val_loss': training_report['training_summary']['final_val_loss'],
            'model_size_mb': _estimate_model_size(model)
        }
    }
    
    logger.info(f" Modèle sauvegardé")
    logger.info(f"   Taille: {save_report['model_info']['model_size_mb']:.1f} MB")
    
    return save_report

# Fonctions utilitaires

def _create_dummy_batch(config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Crée un batch dummy pour test"""
    return {
        'node_embeddings': torch.randn(20, config['node_dim']),
        'edge_embeddings': torch.randn(30, config['edge_dim']),
        'edge_index': torch.randint(0, 20, (2, 30)),
        'node_types': torch.randint(0, 3, (20,)),
        'timestamps': torch.randn(3),
        'shared_node_info': {
            'node_ids': ['test_ip', 'test_service'],
            'embeddings': torch.randn(2, config['node_dim']),
            'packet_associations': [[0, 1], [1, 2]]
        }
    }

def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Déplace batch sur device"""
    moved_batch = {}
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        elif isinstance(value, dict):
            moved_batch[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, torch.Tensor):
                    moved_batch[key][subkey] = subvalue.to(device)
                else:
                    moved_batch[key][subkey] = subvalue
        else:
            moved_batch[key] = value
    
    return moved_batch

def _estimate_model_size(model: torch.nn.Module) -> float:
    """Estime taille modèle en MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(buf.numel() * buf.element_size() for buf in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)