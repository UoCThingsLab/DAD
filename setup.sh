#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-user=sepehr.sabour@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

#Update code
#git fetch
#
##Setup dataset
#rm -rf dataset
#mkdir dataset
#cd dataset
#kaggle datasets download -d pesehr/driving-anomaly-detection
#mkdir v0.1
#unzip driving-anomaly-detection.zip -d ./v0.1
#rm driving-anomaly-detection.zip
#cd ..
#
##Setup conda
#conda create --name DAD --file requirements.txt

#Run 
python3 main.py
