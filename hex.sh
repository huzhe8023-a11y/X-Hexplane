#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --partition=sljus
#SBATCH --output=logs/%j-%x.out
#SBATCH --job-name=strt
#SBATCH --mem-per-cpu=49G   # memory per cpu-core
echo -n "This script is running on "
hostname

module load Anaconda3
#module load GCCcore/9.2.0
source activate /home/zhehu123/.conda/envs/hexplane
cd /data/staff/tomograms/users/zhehu/Hexplane_rotation

/home/zhehu123/.conda/envs/hexplane/bin/python3 main.py config=XMPI.yaml





