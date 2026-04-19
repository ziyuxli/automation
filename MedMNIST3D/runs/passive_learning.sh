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

cd ..

python passive_learning.py --data_flag fracturemnist3d --download --gpu_ids 0 --conv Conv3d --output_root ./output_passive --as_rgb --shape_transform --batch_size 32 --model_flag resnet50 --run passive-model1 --samples_per_round 20 --max_epochs 10 --initial_size 200
# python passive_learning.py --data_flag adrenalmnist3d --download --gpu_ids 0 --conv Conv3d --output_root ./output_passive --as_rgb --shape_transform --batch_size 32 --model_flag resnet50 --run passive-model1 --samples_per_round 20 --max_epochs 10 --initial_size 200


# python train_and_eval_pytorch.py \
#   --data_flag organmnist3d \
#   --download \
#   --gpu_ids 0 \
#   --conv Conv3d \
#   --output_root ./output_passive \
#   --num_epochs 40 \
#   --as_rgb \
#   --shape_transform \
#   --batch_size 32 \
#   --model_flag resnet50 \
#   --run model1

# python passive_learning.py \
#   --data_flag vesselmnist3d \
#   --output_root ./output_passive \
#   --num_epochs 20 \
#   --conv ACSConv \
#   --model_flag resnet18 \
#   --imagenet_pretrained \
#   --passive_initial_ratio 0.5 \
#   --passive_add_ratio 0.05 \
#   --passive_update_every 1 \
#   --download \
#   --as_rgb
