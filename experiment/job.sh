#!/bin/sh

#SBATCH --account=cs-dclabs-2019
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

cd ../../
source venv/bin/activate
cd Generalized-RL-Self-Play-Framework/experiment/
python run.py --config experiment_config_robosumo_minimal.yaml --dest ./

