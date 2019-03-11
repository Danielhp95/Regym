#!/bin/sh

#SBATCH --account=cs-dclabs-2019
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
cd ../../../
source venv/bin/activate
cd Generalized-RL-Self-Play-Framework/tests/environments
pytest -s robosumowithrewardshaping_ppo_multiactor_test.py 
