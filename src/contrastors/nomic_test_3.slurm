#! /bin/sh

#SBATCH --job-name=nomic3_finetune
#SBATCH --output=/home/sharifm/students/yorayh/nomic3_finetune.out # redirect stdout
#SBATCH --error=/home/sharifm/students/yorayh/nomic3_finetune.err  # redirect stderr
#SBATCH --partition=gpu-sharifm # (see resources section)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 #number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total
#SBATCH --nodelist=n-804

torchrun --nproc-per-node=1 --master_port 31338 train.py --config=configs/train/contrastive_finetune_3.yaml --dtype=bf16 
