MODEL_SIZE="35"
MODEL_ID=llamacpp/qwen3.5-${MODEL_SIZE}b
BASE_OUTPUT_PATH="../logs/frames/llamacpp_qwen3.5_${MODEL_SIZE}b"
mkdir -p ${BASE_OUTPUT_PATH}

sudo nvpmodel -m 0
sudo nvpmodel -q
sudo jetson_clocks
sudo jetson_clocks --show

sudo echo "Starting profiling"
nohup python -u -m agentstop.profiler.profile_and_analyze_batch \
--script_template "python -m agentstop.agents.run \
    --agent_type web_logprobs \
    --model_id openai/${MODEL_ID} \
    --model_type litellm \
    --api_base http://127.0.0.1:8080/v1 \
    --prompt {prompt} \
    --temperature 0.7 \
    --top_p 0.8 \
    --min_p 0.0 \
    --top_k 20 \
    --max_tokens 2048 \
    --stream \
    --trace_path {agent_trace_path}" \
--input_path ../experiment_data/processed/frames/llm_frames_results_judged.csv \
--question_col Prompt \
--num_repeats 1 \
--num_retries 1 \
--timeout 900 \
--base_output_path ${BASE_OUTPUT_PATH} \
--preload_model_id ${MODEL_ID} \
--llm_backend LlamaCpp \
--device_type jetson \
> ${BASE_OUTPUT_PATH}/profile.log 2>&1 &
