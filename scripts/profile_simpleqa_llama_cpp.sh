MODEL_SIZE="30"
MODEL_ID=llamacpp/qwen3-${MODEL_SIZE}b
BASE_OUTPUT_PATH="../logs/simpleqa/llamacpp_qwen3_${MODEL_SIZE}b"
mkdir -p ${BASE_OUTPUT_PATH}
sudo echo "Starting profiling"
sudo nohup python -u -m agentstop.profiler.profile_and_analyze_batch \
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
    --stream \
    --trace_path {agent_trace_path}" \
--input_path ../experiment_data/processed/simpleqa/llm_simpleqa_results_qwen3_30b_2507_judged.csv \
--question_col problem \
--num_repeats 1 \
--num_retries 1 \
--timeout 300 \
--base_output_path ${BASE_OUTPUT_PATH} \
--preload_model_id ${MODEL_ID} \
--llm_backend LlamaCpp \
--device_type apple_laptop \
> ${BASE_OUTPUT_PATH}/profile.log 2>&1 &

caffeinate -dimsu tail -n 100 -f ${BASE_OUTPUT_PATH}/profile.log