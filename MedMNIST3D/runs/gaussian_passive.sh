#!/bin/bash
#SBATCH --job-name=model_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=../slurm_out/sbatch_output/output-%A-%x-%u.out
#SBATCH --time=8-2:00:00
#SBATCH --error=../slurm_out/sbatch_error/slurm_error_%j.log
#SBATCH --nodelist=mind-0-28


echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/.bashrc
conda activate automation

cd ..

# python passive_learning.py --data_flag fracturemnist3d --download --gpu_ids 0 --conv Conv3d --output_root ./output_passive --as_rgb --shape_transform --batch_size 32 --model_flag resnet50 --run passive-model1 --samples_per_round 20 --max_epochs 10 --initial_size 200
# python passive_learning.py \
#     --data_flag fracturemnist3d \
#     --download \
#     --gpu_ids 0 \
#     --conv Conv3d \
#     --output_root ./output_passive \
#     --as_rgb \
#     --shape_transform \
#     --batch_size 32 \
#     --model_flag resnet50 \
#     --run passive-model1 \
#     --samples_per_round 20 \
#     --max_epochs 10 \
#     --initial_size 200

python gaussian_passive.py \
    --data_flag fracturemnist3d \
    --output_root ./output_gaussian \
    --samples_per_round 20 \
    --max_epochs 10 \
    --gpu_ids 0 \
    --batch_size 32 \
    --conv Conv3d \
    --download \
    --model_flag resnet50 \
    --as_rgb \
    --shape_transform \
    --run passive-model1 \
    --initial_size 200 \
    --strategy entropy \
    --gaussian_drop_prob 0.2 \
    --mc_samples 20 \