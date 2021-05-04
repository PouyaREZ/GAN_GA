#!/bin/bash
#SBATCH --job-name=RQ3_EnergyNetworkOnly
#SBATCH --output=RQ3_EnergyNetworkOnly.%j.out
#SBATCH --error=RQ3_EnergyNetworkOnly.%j.err
#SBATCH --time=1-00:00:00
#SBATCH -p cee
#SBATCH -c 20
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pouyar@stanford.edu
module load python/3.6.1
module load gurobi/9.0.3_py36
export GRB_LICENSE_FILE=/share/software/user/restricted/gurobi/9.0.3_py36/licenses/gurobi.lic
python3.6 -m scoop --host $HOSTNAME -vv -n 20 RQ3.py

