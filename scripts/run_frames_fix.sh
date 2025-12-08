# export OLLAMA_CONTEXT_LENGTH=40960
# export OLLAMA_KV_CACHE_TYPE=f16
# export OLLAMA_FLASH_ATTENTION=1
# nohup ollama serve > ../logs/ollama_ctx_40960_kvq_f16.log 2>&1 &
# sleep 2
python -m efficient_agents.eval.eval_smolagents_qa_fix \
--model_id qwen3:1.7b \
--compression_ratio 1.0 \
--input_path ../data/frames/llm_frames_results_all_ctx_40960_kvq_f16_fixed.csv \
--question_col Prompt \
--answer_col Answer \
--output_path ../data/frames/llm_frames_results_all_ctx_40960_kvq_f16_fixed.csv \
--output_col_pattern code_{0}_ctx_40960_kvq_f16 > ../logs/llm_frames_results_all_ctx_40960_kvq_f16_fix_3.log 2>&1 &
