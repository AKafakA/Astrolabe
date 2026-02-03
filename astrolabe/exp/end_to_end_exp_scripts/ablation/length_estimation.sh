# The host with GPU to run the length estimation script
GPU_HOST = ""
ASTROLABE_GITHUB_LINK="https://github.com/anonymous/vllm-astrolabe.git"

parallel-ssh -t 0 -h GPU_HOST "pip install -U pip==25.0.1"
parallel-ssh -t 0 -h GPU_HOST "pip install accelerate deepspeed einops fschat peft simpletransformers fsspec==2025.3.2"
parallel-ssh -t 0 -h GPU_HOST "git clone ${ASTROLABE_GITHUB_LINK} "
parallel-ssh -t 0 -h GPU_HOST "cd Astrolabe && python astrolabe/length_estimation/train_roberta.py"
parallel-ssh -t 0 -h GPU_HOST "cd Astrolabe && python astrolabe/length_estimation/eval_roberta.py"