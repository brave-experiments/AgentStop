MODEL_SIZE="30" # Initial model
MODEL2_SIZE="30" # Model to cascade to. Ignore this
THRESHOLD=0 # Setting to 0 will disable cascading

MODEL_ID=llamacpp/qwen3-${MODEL_SIZE}b
MODEL2_ID=llamacpp/qwen3-${MODEL2_SIZE}b
BASE_OUTPUT_PATH="../logs/frames_full_llamacpp_qwen3_${MODEL_SIZE}b"
mkdir -p ${BASE_OUTPUT_PATH}

sudo echo "Starting profiling"
sudo nohup python -u -m efficient_agents.profiler.profile_and_analyze_batch \
--script_template "python -m efficient_agents.agents.web.smol_agents \
    --agent_type logprobs_cascade \
    --model_id openai/${MODEL_ID} openai/${MODEL2_ID} \
    --model_type litellm \
    --api_base http://127.0.0.1:8080/v1 \
    --prompt {prompt} \
    --temperature 0.7 \
    --top_p 0.8 \
    --min_p 0.0 \
    --top_k 20 \
    --logprobs_threshold ${THRESHOLD} \
    --stream \
    --trace_path {agent_trace_path}" \
--input_path ../data/frames/llm_frames_results_judged.csv \
--question_col Prompt \
--num_repeats 1 \
--num_retries 1 \
--timeout 300 \
--base_output_path ${BASE_OUTPUT_PATH} \
--preload_model_id ${MODEL_ID} \
--llm_backend LlamaCpp \
--device_type apple_laptop \
> ${BASE_OUTPUT_PATH}/profile.log 2>&1 &
