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

python3 main.py -H 8 -l 10 -d 0.9
python3 main.py -H 10 -l 10 -d 0.9
python3 main.py -H 12 -l 10 -d 0.9
python3 main.py -H 14 -l 10 -d 0.9
python3 main.py -H 16 -l 10 -d 0.9
python3 main.py -H 10 -l 1 -d 0.9
python3 main.py -H 10 -l 0.1 -d 0.9
python3 main.py -H 16 -l 10 -d 0.11
