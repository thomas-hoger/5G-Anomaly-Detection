#!/bin/bash

#SBATCH --job-name=kedro_gpu # Nom du job
#SBATCH --output=kedro_gpu_%j.out # Fichier de sortie (%j = ID du job)
#SBATCH --error=kedro_gpu_%j.err # Fichier d'erreur
#SBATCH --ntasks=1 # Nombre de tâches
#SBATCH --nodes=1 # Nombre de nœuds
#SBATCH --gres=gpu:1 # Demande 1 GPU
#SBATCH --partition=gpu # Partition GPU
#SBATCH --mem=190G
#SBATCH --time=144:00:00 # Temps limite (2 heures)

Affichage des informations du job
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

Ajouter ~/.local/bin au PATH pour éviter les warnings
export PATH=$HOME/.local/bin:$PATH

Vérification de Python
python3 --version || { echo "Python3 non trouvé !"; exit 1; }

Mise à jour de pip et correction des dépendances
pip install --upgrade pip
pip install packaging # Correction du problème de version
pip install -r requirements.txt # Ajout de Kedro-Viz

Vérification de l'environnement GPU
nvidia-smi

python -c "import torch;
print('PyTorch version:', torch.version);
print('CUDA available:', torch.cuda.is_available());
print('CUDA device count:', torch.cuda.device_count());
print('CUDA device name:', torch.cuda.get_device_name(0))"

Exécution de Kedro
cd /pfcalcul/work/mamdouni/first_simulation # Remplace par ton chemin réel
kedro run --pipeline gnn_pretraining

Fin du job
echo "Job terminé à $(date)"