MODEL_SIZE="1.7"
MODEL2_SIZE="14"
BASE_OUTPUT_PATH="../logs/frames_mlc_qwen3_mean_log_probs_cascade_compress_${MODEL_SIZE}b_to_${MODEL2_SIZE}b"
mkdir -p ${BASE_OUTPUT_PATH}
sudo echo "Starting profiling"
# mkdir -p ../logs/frames_ollama_qwen3_${MODEL_SIZE}b
# sudo echo "Starting profiling"
# sudo nohup python -u -m efficient_agents.profiler.profile_and_analyze_batch \
# --script_template "python -m efficient_agents.agents.web.smol_agents \
#     --agent_type code \
#     --model_id ollama_chat/qwen3:${MODEL_SIZE}b \
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
# --base_output_path ../logs/frames_ollama_qwen3_${MODEL_SIZE}b \
# --preload_model_id ollama_chat/qwen3:${MODEL_SIZE}b \
# --device_type apple_laptop \
# > ../logs/frames_ollama_qwen3_${MODEL_SIZE}b/profile.log 2>&1 &
sudo nohup python -u -m pefficient_agents.rofiler.profile_and_analyze_batch \
--script_template "python -m efficient_agents.agents.web.smol_agents \
    --agent_type mean_log_probs \
    --model_id openai/mlc-ai/Qwen3-${MODEL_SIZE}B-q4f16_1-MLC openai/mlc-ai/Qwen3-${MODEL2_SIZE}B-q4f16_1-MLC \
    --model_type litellm \
    --api_base http://127.0.0.1:8080/v1 \
    --llm_backend Apple_MLC_Server \
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
--preload_model_id mlc-ai/Qwen3-${MODEL_SIZE}B-q4f16_1-MLC \
--llm_backend Apple_MLC_Server \
--device_type apple_laptop \
> ${BASE_OUTPUT_PATH}/profile.log 2>&1 &