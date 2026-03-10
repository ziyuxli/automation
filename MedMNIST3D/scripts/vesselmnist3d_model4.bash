#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 03:30:00
#SBATCH --gpus=v100-32:1
#SBATCH -o /ocean/projects/cis250072p/wanyuef/asu/experiments/MedMNIST3D/slurm_out/try_%j.out

#type 'man sbatch' for more information and options
#this job will ask for 1 full v100-32 GPU node(8 V100 GPUs) for 5 hours
#this job would potentially charge 40 GPU SUs

#echo commands to stdout
set -x

# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - PSC-username should be replaced by your PSC username
# - path-to-directory should be replaced by the path to your directory where the executable is

export HF_HOME=/ocean/projects/cis250072p/wanyuef/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub   

mkdir -p "$HF_HUB_CACHE"

huggingface-cli scan-cache --dir ~/.cache/huggingface 2>/dev/null || true

export BIGCACHE=/ocean/projects/cis250072p/wanyuef/.big-cache

export XDG_CACHE_HOME=$BIGCACHE/xdg
mkdir -p "$XDG_CACHE_HOME"

export TORCHINDUCTOR_CACHE_DIR=$BIGCACHE/torchinductor
export TRITON_CACHE_DIR=$BIGCACHE/triton
export VLLM_TORCH_COMPILE_CACHE_DIR=$BIGCACHE/vllm/torch_compile_cache
export CUDA_CACHE_PATH=$BIGCACHE/cuda 
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_TORCH_COMPILE_CACHE_DIR" "$CUDA_CACHE_PATH"

export VLLM_TORCH_COMPILE_CACHE_DIR=$BIGCACHE/vllm/torch_compile_cache
mkdir -p "$VLLM_TORCH_COMPILE_CACHE_DIR"

cd /ocean/projects/cis250072p/wanyuef/asu/experiments/MedMNIST3D

module load anaconda3
conda activate /ocean/projects/cis250072p/wanyuef/conda-envs/medmnist

ALL_NODES=($(scontrol show hostnames $SLURM_NODELIST))

COMMANDS=(
"python train_and_eval_pytorch.py --data_flag vesselmnist3d --download --gpu_ids 0 --conv Conv3d --output_root /ocean/projects/cis250072p/wanyuef/asu/output --num_epochs 100 --as_rgb --shape_transform --batch_size 32 --model_flag resnet50 --run model4"
)

for i in "${!COMMANDS[@]}"; do
    start_idx=$((i*8))
    end_idx=$((start_idx+8))
    GROUP_NODES=("${ALL_NODES[@]:$start_idx:8}") 
    GROUP_NODELIST=$(IFS=, ; echo "${GROUP_NODES[*]}")   
    log_file="logs/debug_${SLURM_JOB_ID}_$i.log"
    {
        echo "Command: ${COMMANDS[$i]}"
        node_offset=$((i * 8))
        srun --nodes=1 --ntasks=1 --ntasks-per-node=1 -w $GROUP_NODELIST\
             --exclusive --exact --cpu-bind=cores \
             bash -c "
	     export MASTER_ADDR=${GROUP_NODES[0]}
             export MASTER_PORT=$((29505 + i))
             ${COMMANDS[$i]}"
    } > "$log_file" 2>&1 &
done

wait

