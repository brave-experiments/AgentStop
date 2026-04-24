MODEL_ID=llamacpp/qwen3-coder-30b
BASE_OUTPUT_PATH="../logs/swebench/llamacpp_qwen3_coder_30b"
mkdir -p ${BASE_OUTPUT_PATH}

# First time run should uncomment this
# colima stop
# colima start --cpu 8 --memory 16 --disk 120

sudo echo "Starting profiling"
sudo nohup python -u -m efficient_agents.profiler.profile_and_analyze_swebench \
--script_template "python -m efficient_agents.agents.run \
    --agent_type swebench_logprobs \
    --docker_id {docker_id} \
    --model_id openai/${MODEL_ID} \
    --model_type litellm \
    --api_base http://127.0.0.1:8080/v1 \
    --prompt_path {prompt_path} \
    --temperature 0.7 \
    --top_p 0.8 \
    --min_p 0.0 \
    --top_k 20 \
    --max_steps 100 \
    --max_tokens 4096 \
    --stream \
    --trace_path {agent_trace_path}" \
--input_path "hf://datasets/SWE-bench/SWE-bench_Verified/data/test-00000-of-00001.parquet" \
--question_col problem_statement \
--num_repeats 1 \
--num_retries 1 \
--timeout 7200 \
--base_output_path ${BASE_OUTPUT_PATH} \
--preload_model_id ${MODEL_ID} \
--llm_backend LlamaCpp \
--device_type apple_laptop \
> ${BASE_OUTPUT_PATH}/profile.log 2>&1 &

