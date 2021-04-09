#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem=2G
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-user=sepehr.sabour@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

python3 main.py -H 12 -l 10 -d 0.9
