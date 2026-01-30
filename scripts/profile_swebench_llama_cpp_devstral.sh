THRESHOLD=0 # Setting to 0 will disable cascading

MODEL_ID=llamacpp/devstral-small-2
BASE_OUTPUT_PATH="../logs/swebench/llamacpp_devstral_small_2"
mkdir -p ${BASE_OUTPUT_PATH}

# First time run should uncomment this
# colima stop
# colima start --cpu 8 --memory 16 --disk 120

sudo echo "Starting profiling"
sudo nohup python -u -m efficient_agents.profiler.profile_and_analyze_swebench \
--script_template "python -m efficient_agents.agents.run \
    --agent_type swebench_logprobs \
    --docker_id {docker_id} \
    --model_id openai/${MODEL_ID} openai/${MODEL_ID} \
    --model_type litellm \
    --api_base http://127.0.0.1:8080/v1 \
    --prompt_path {prompt_path} \
    --temperature 0.15 \
    --min_p 0.01 \
    --max_steps 100 \
    --logprobs_threshold ${THRESHOLD} \
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

