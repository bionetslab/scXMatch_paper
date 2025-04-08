#!/bin/bash -l 
#SBATCH -J nlof
#SBATCH --nodes=1                  
#SBATCH --ntasks=1           
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --export=NONE      
#SBATCH --time=12:00:00

module add python
conda activate gt
cd /home/woody/iwbn/iwbn007h/scXMatch_paper/src
PYTHONNOUSERSITE=1 python nlof.py

