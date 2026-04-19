#!/bin/bash
#SBATCH --job-name=model_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out
#SBATCH --time=8-2:00:00
#SBATCH --error=./sbatch_error/slurm_error_%j.log
#SBATCH --nodelist=mind-0-28


echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/.bashrc
conda activate automation

cd ../MedMNIST3D

python passive_learning.py \
--data_flag fracturemnist3d \
--download \
--num_epochs 50 \
--model_flag resnet18 \
--conv ACSConv \
--imagenet_pretrained \
--as_rgb
