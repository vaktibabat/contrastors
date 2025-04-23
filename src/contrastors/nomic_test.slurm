#! /bin/sh

#SBATCH --job-name=nomic_finetune
#SBATCH --output=/home/sharifm/students/yorayh/nomic_finetune.out # redirect stdout
#SBATCH --error=/home/sharifm/students/yorayh/nomic_finetune.err  # redirect stderr
#SBATCH --partition=gpu-sharifm # (see resources section)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 #number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total

torchrun --nproc-per-node=1 train.py --config=configs/train/contrastive_finetune.yaml --dtype=bf16 
