parallel-ssh -i -t 0 -h astrolabe/config/hosts "cd Astrolabe && rm -rf experiment_output/logs/* && mkdir -p experiment_output/logs"

parallel-ssh -h astrolabe/config/hosts "pkill -f vllm.entrypoints.api_server"
parallel-ssh -h astrolabe/config/hosts "pkill -f predictor"
parallel-ssh -h astrolabe/config/hosts "pkill -f multiprocessing.spawn"