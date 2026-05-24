环境配置：
conda create -n lerobot-lambda -c conda-forge --override-channels python=3.12 -y

python -m pip install -U pip setuptools wheel

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[smolvla,dataset,dev,scipy-dep]"

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

数据集
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='HuggingFaceVLA/libero', repo_type='dataset', local_dir=r'D:\huggingface\hf_cache\lerobot\HuggingFaceVLA\libero', local_dir_use_symlinks=False, endpoint='https://hf-mirror.com', resume_download=True)"

vlm模型
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='HuggingFaceTB/SmolVLM2-500M-Video-Instruct', repo_type='model', local_dir=r'D:\huggingface\hf_cache\models\SmolVLM2-500M-Video-Instruct', local_dir_use_symlinks=False, endpoint='https://hf-mirror.com', resume_download=True)"

标签生成
python -m lerobot.policies.smolvla.lambda_labels ^
  --repo-id=HuggingFaceVLA/libero ^
  --root=D:\huggingface\hf_cache\lerobot\HuggingFaceVLA\libero ^
  --output-path=D:\huggingface\hf_cache\lerobot\HuggingFaceVLA\libero\lambda_labels.pt ^
  --lambda-error-q-low=0.03 ^
  --lambda-error-q-high=0.97 ^
  --lambda-envelope-window=7 ^
  --lambda-smoothing-window=15 ^
  --lambda-smoothing-alpha=0.6 ^
  --diagnostics-path=D:\huggingface\hf_cache\lerobot\HuggingFaceVLA\libero\lambda_diagnostics.csv

python -m lerobot.policies.smolvla.lambda_label_viewer ^
  --repo-id=HuggingFaceVLA/libero ^
  --root=D:\huggingface\hf_cache\lerobot\HuggingFaceVLA\libero ^
  --labels-path=D:\huggingface\hf_cache\lerobot\HuggingFaceVLA\libero\lambda_labels_C.pt ^
  --episode=0 ^
  --fps=10 ^
  --output-video=D:\huggingface\hf_cache\lerobot\lambda_videos\lambda_episode0.mp4

lambda预测头训练：
缓存平均池化后的prefix_hidden
python -m lerobot.policies.smolvla.debug_lambda_only \
  --root=/root/autodl-tmp/hf_cache/huggingface/lerobot/HuggingFaceVLA/libero \
  --vlm-model-name=/root/autodl-tmp/hf_cache/huggingface/models/lerobot/SmolVLM2-500M-Video-Instruct \
  --device=cuda \
  --cache-mode=save \
  --cache-path=/root/autodl-tmp/hf_cache/huggingface/lerobot/vlm_hidden_cache.pt \
  --cache-batch-size=512 \
  --use-image-cache \
  --image-cache-dir=/root/autodl-tmp/image_cache/libero

# 使用全局池化
python -m lerobot.policies.smolvla.debug_lambda_only ^
  --root=D:\huggingface\hf_cache\lerobot\HuggingFaceVLA\libero ^
  --labels-path=D:\huggingface\hf_cache\lerobot\HuggingFaceVLA\libero\lambda_labels_C.pt ^
  --vlm-model-name=D:\huggingface\hf_cache\models\SmolVLM2-500M-Video-Instruct ^
  --device=cuda ^
  --steps=273465 ^
  --batch-size=64 ^
  --lambda-num-bins=11 ^
  --cache-mode=load ^
  --cache-path=D:\huggingface\hf_cache\lerobot\vlm_hidden_cache.pt.shards ^
  --distance-weight=1.0 ^
  --distance-power=2.0 ^
  --variance-weight=200 ^
  --lr=2e-3 ^
  --vis-max-frames=1000 ^
  --vis-output-dir=D:\huggingface\hf_cache\lerobot\lambda_debug_videos ^
  --test-episodes=0,1,2,3 ^
  --save-head-path=D:\huggingface\hf_cache\lerobot\lambda_debug_head.pt
