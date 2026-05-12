一、环境准备：
1.1 安装依赖
conda create -y -n lerobot python=3.12
conda activate lerobot

conda install -y -c conda-forge ffmpeg=7.1.1

python -m pip install -U pip setuptools wheel
python -m pip install -i https://pypi.org/simple num2words==0.5.14

git clone https://github.com/huggingface/lerobot.git
git clone -b feat/dataset-read-cache --single-branch https://github.com/whw0811/lerobot.git

pip install -e ".[smolvla,libero]"

hf auth login

wandb login

1.2 配置环境变量
export HF_HOME=/root/autodl-tmp/hf_cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/huggingface/datasets
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/huggingface/hub
export TMPDIR=/root/autodl-tmp/tmp
export HF_HUB_DISABLE_XET=1
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p $HF_HOME
mkdir -p $HF_DATASETS_CACHE
mkdir -p $HF_HUB_CACHE
mkdir -p $TMPDIR

1.3 数据集下载
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HuggingFaceVLA/libero",
    repo_type="dataset",
    local_dir="/root/autodl-tmp/hf_cache/huggingface/lerobot/HuggingFaceVLA/libero",
    local_dir_use_symlinks=False,
    endpoint="https://hf-mirror.com",
    resume_download=True,
)
PY

1.4 下载并放置 LIBERO 资源包
mkdir -p /root/autodl-tmp/libero_assets_download
mkdir -p /root/miniconda3/envs/lerobot/lib/python3.12/site-packages/libero/libero/assets

hf download lerobot/libero-assets \
  --repo-type dataset \
  --local-dir /root/autodl-tmp/libero_assets_download

cp -r /root/autodl-tmp/libero_assets_download/* \
/root/miniconda3/envs/lerobot/lib/python3.12/site-packages/libero/libero/assets/

1.5 下载模型
1.5.1
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lerobot/smolvla_base",
    repo_type="model",
    local_dir="/root/autodl-tmp/hf_cache/huggingface/models/lerobot/smolvla_base",
    local_dir_use_symlinks=False,
    endpoint="https://hf-mirror.com",
    resume_download=True,
)
PY
1.5.2
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HuggingFaceVLA/smolvla_libero",
    repo_type="model",
    local_dir="/root/autodl-tmp/hf_cache/huggingface/models/lerobot/smolvla_libero",
    local_dir_use_symlinks=False,
    endpoint="https://hf-mirror.com",
    resume_download=True,
)
PY
1.5.3
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    repo_type="model",
    local_dir="/root/autodl-tmp/hf_cache/huggingface/models/lerobot/SmolVLM2-500M-Video-Instruct",
    local_dir_use_symlinks=False,
    endpoint="https://hf-mirror.com",
    resume_download=True,
)
PY

二、训练
2.1 离线 residual predictor 训练 + λ 标签生成
python -m lerobot.policies.smolvla.lambda_labels \
  --repo-id=HuggingFaceVLA/libero \
  --root=/root/autodl-tmp/hf_cache/huggingface/lerobot/HuggingFaceVLA/libero \
  --output-path=/root/autodl-tmp/hf_cache/huggingface/lerobot/HuggingFaceVLA/libero/lambda_labels.pt \
  --lambda-error-q-low=0.05 \
  --lambda-error-q-high=0.95 \
  --lambda-envelope-window=5 \
  --lambda-smoothing-window=11 \
  --lambda-smoothing-alpha=0.7 \
  --diagnostics-path=/root/autodl-tmp/hf_cache/huggingface/lerobot/HuggingFaceVLA/libero/lambda_diagnostics.csv

  2.2 正式训练
lerobot-train \
  --policy.type=smolvla \
  --policy.vlm_model_name=/root/autodl-tmp/hf_cache/huggingface/models/lerobot/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=true \
  --policy.device=cuda \
  --policy.num_vlm_layers=16 \
  --policy.n_obs_steps=10 \
  --policy.push_to_hub=false \
  --policy.lambda_labels_path=/root/autodl-tmp/hf_cache/huggingface/lerobot/HuggingFaceVLA/libero/lambda_labels.pt \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root=/root/autodl-tmp/hf_cache/huggingface/lerobot/HuggingFaceVLA/libero \
  --batch_size=64 \
  --num_workers=4 \
  --steps=100000 \
  --save_freq=5000 \
  --seed=42 \
  --wandb.enable=false \
  --eval_freq=0 \
  --dataset.use_image_cache=true \
  --dataset.image_cache_dir=/root/autodl-tmp/image_cache/libero \
  --output_dir=/root/autodl-tmp/outputs/train/smolvla_vlm_lambda

2.3 测试
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

lerobot-eval \
  --output_dir=/root/autodl-tmp/outputs/eval/smolvla_vlm_lambda/1to50_dynamic_freq/100000 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.path=/root/autodl-tmp/outputs/train/smolvla_vlm_lambda/checkpoints/100000/pretrained_model \
  --policy.n_action_steps=10 \
  --seed=42 \
  --policy.dynamic_n_action_steps=true \
  --policy.dynamic_n_action_steps_min=5 \
  --policy.dynamic_n_action_steps_max=10