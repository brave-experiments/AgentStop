export OLLAMA_CONTEXT_LENGTH=40960
export OLLAMA_KV_CACHE_TYPE=f16
export OLLAMA_FLASH_ATTENTION=1
nohup ollama serve > ../logs/ollama_ctx_40960_kvq_f16.log 2>&1 &
sleep 2
nohup python -m efficient_agents.eval.eval_smolagents_qa \
--model_id qwen3:30b qwen3:32b \
--compression_ratio 1.0 \
--input_path ../data/simpleqa/agents_simpleqa_results_ctx_40960_kvq_f16_plan_3_judged.csv \
--question_col problem \
--answer_col answer \
--output_path ../data/simpleqa/simpleqa_results_all_ctx_40960_kvq_f16.csv \
--output_col_pattern code_{0}_ctx_40960_kvq_f16 > ../logs/simpleqa_results_all_ctx_40960_kvq_f16.log 2>&1 &
