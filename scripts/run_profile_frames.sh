export OLLAMA_CONTEXT_LENGTH=40960
export OLLAMA_KV_CACHE_TYPE=f16
export OLLAMA_FLASH_ATTENTION=1
nohup ollama serve > /dev/null 2>&1 &
sleep 2
MODEL_SIZE="1.7"
MODEL2_SIZE="30"
BASE_OUTPUT_PATH=../logs/frames_ollama_qwen3_${MODEL_SIZE}b
mkdir -p ${BASE_OUTPUT_PATH}
sudo echo "Starting profiling"
sudo nohup python -u -m efficient_agents.profiler.profile_and_analyze_batch \
--script_template "python -m efficient_agents.agents.run \
    --agent_type web_basic \
    --model_id ollama_chat/qwen3:${MODEL_SIZE}b \
    --model_type litellm \
    --prompt {prompt} \
    --temperature 0.7 \
    --top_p 0.8 \
    --min_p 0.0 \
    --top_k 20 \
    --stream \
    --trace_path {agent_trace_path}" \
--input_path ../data/frames/llm_frames_results_all_ctx_40960_kvq_f16.csv \
--question_col Prompt \
--num_repeats 1 \
--base_output_path ${BASE_OUTPUT_PATH} \
--preload_model_id qwen3:${MODEL_SIZE}b \
--llm_backend Ollama \
--device_type apple_laptop \
> ${BASE_OUTPUT_PATH}/profile.log 2>&1 &
# sudo nohup python -u -m efficient_agents.profiler.profile_and_analyze_batch \
# --script_template "python -m efficient_agents.agents.run \
#     --agent_type web_basic \
#     --fixed_cascade_step 2 \
#     --model_id ollama_chat/qwen3:1.7b ollama_chat/qwen3:30b-a3b-instruct-2507-q4_K_M \
#     --model_type litellm \
#     --prompt {prompt} \
#     --temperature 0.7 \
#     --top_p 0.8 \
#     --min_p 0.0 \
#     --top_k 20 \
#     --stream \
#     --trace_path {agent_trace_path}" \
# --input_path ../data/frames/llm_frames_results_all_ctx_40960_kvq_f16.csv \
# --question_col Prompt \
# --num_repeats 1 \
# --base_output_path ../logs/frames_ollama_qwen3_fixed_cascade_compress_1.7b_to_30b_step_2 \
# --preload_model_id ollama_chat/qwen3:1.7b \
# --device_type apple_laptop \
# > ../logs/frames_ollama_qwen3_fixed_cascade_compress_1.7b_to_30b_step_2/profile.log 2>&1 &
