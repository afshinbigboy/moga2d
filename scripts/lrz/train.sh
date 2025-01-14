#!/bin/bash
#SBATCH -N 1
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:1
#SBATCH --job-name=mogaSy
#SBATCH -t 1-00:00:00
#SBATCH --mail-user=afshinbigboy@gmail.com
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --container-image="/dss/dssfs04/lwp-dss-0002/pn36fu/pn36fu-dss-0000/env/pytorch-env.sqsh"
#SBATCH --container-mounts="/dss/dssfs04/lwp-dss-0002/pn36fu/pn36fu-dss-0000:/cabinet"


# Define the section to execute
SECTION=$1
if [ -z "$SECTION" ]; then
  echo "Error: No section specified."
  echo "Valid sections are: SYNAPSE"
  exit 1
fi
# Define the fold variable, defaulting to 0 if not provided
fold=${2:-0}

echo "SECTION: $SECTION"
nvidia-smi


# cd 
# Load the environment variables
# ./env_v2.sh

base_path="${HOME}/deeplearning/sina/moga2d"
run_path="${base_path}/train_xmind_bou.py"
label="moga_cat_baseline"

case "$SECTION" in
  SYNAPSE)
    echo "Running SYNAPSE-2D section..."
    cd ${base_path}

    python -m pip install --upgrade pip
    pip install -r requirements.txt

    pip install opencv-python==4.7.0.72

    echo "Fixing required files"
    cp files_tobereplaced/binary.py /usr/local/lib/python3.10/dist-packages/medpy/metric/binary.py
    cp files_tobereplaced/meta.py /usr/local/lib/python3.10/dist-packages/imgaug/augmenters/meta.py

    python ${run_path} \
      --eval_interval 15 \
      --max_epochs 450 \
      --root_path /cabinet/dataset/Synapse/train_npz \
      --test_path /cabinet/dataset/Synapse/test_vol_h5 \
      --output_dir '/cabinet/reza/sina/miccai2025/test-moga-af-boundaryskip-main-0.60.3' \
      --batch_size 16 \
      --num_workers 4 \
      --model_name mogav5dwskip \
      --optimizer 'SGD' \
      --base_lr 0.05 \
      --scale_factors '0.6,0.3' \
      --dice_loss_weight 0.6 \

    ;;
    *)
    echo "Invalid section specified: $SECTION"
    echo "Valid sections are: SYNAPSE"
    exit 1
    ;;
esac



# Namespace(
#   root_path='/cabinet/dataset/Synapse/train_npz', 
#   test_path='/cabinet/dataset/Synapse/test_vol_h5', 
#   dataset='Synapse', 
#   dstr_fast=False, 
#   en_lnum=3, 
#   br_lnum=3, 
#   de_lnum=3, 
#   compact=False, 
#   continue_tr=False, 
#   optimizer='SGD', 
#   dice_loss_weight=0.6, 
#   list_dir='./lists/lists_Synapse', 
#   num_classes=9, 
#   output_dir='/dss/dssfs04/lwp-dss-0002/pn36fu/pn36fu-dss-0000/reza/sina/miccai2025/emcad-without-diff/v5-450-16-afshin-main/mogav5dw', 
#   max_iterations=90000, 
#   max_epochs=450, 
#   batch_size=18, 
#   num_workers=4, 
#   eval_interval=15, 
#   model_name='mogav5dw', 
#   n_gpu=1, 
#   bridge_layers=1, 
#   deterministic=1, 
#   kernel_sizes=[1, 3, 5]
#   expansion_factor=2, 
#   activation_mscb='relu6', 
#   no_dw_parallel=False, 
#   concatenation=False, 
#   encoder='pvt_v2_b2', 
#   no_pretrain=False, 
#   lgag_ks=3, 
#   base_lr=0.05, 
#   img_size=224, 
#   z_spacing=1, 
#   seed=1234, 
#   opts=None, 
#   zip=False, 
#   cache_mode='part', 
#   resume=None, 
#   accumulation_steps=None, 
#   use_checkpoint=False, 
#   amp_opt_level='O1', 
#   tag=None, 
#   eval=False, 
#   throughput=False)
