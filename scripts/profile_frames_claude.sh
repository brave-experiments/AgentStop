MODEL_ID=claude-haiku-4-5-20251001
BASE_OUTPUT_PATH="../logs/frames_full_${MODEL_ID}"
mkdir -p ${BASE_OUTPUT_PATH}

sudo echo "Starting profiling"
sudo nohup python -u -m agentstop.profiler.profile_and_analyze_batch \
--script_template "python -m agentstop.agents.run \
    --agent_type web_logprobs \
    --model_id ${MODEL_ID} \
    --model_type litellm \
    --api_key ANTHROPIC_API_KEY \
    --prompt {prompt} \
    --temperature 0.7 \
    --stream \
    --trace_path {agent_trace_path}" \
--input_path ../experiment_data/processed/frames/llm_frames_results_judged.csv \
--question_col Prompt \
--num_repeats 1 \
--num_retries 1 \
--timeout 300 \
--base_output_path ${BASE_OUTPUT_PATH} \
--device_type apple_laptop \
> ${BASE_OUTPUT_PATH}/profile.log 2>&1 &
