curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "model": "qwen3:1.7b",
        "messages": [
            {"role": "user", "content": "Write a haiku /no_think"}
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "min_p": 0.0,
        "top_k": 20,
        "logprobs": true,
        "top_logprobs": 0
  }' \
  http://127.0.0.1:8080/v1/chat/completions |
  jq -r '
  .choices[0].logprobs.content[] |
  [
    "TOKEN=" + (.token | @json),
    "LOGPROB=" + (.logprob | tostring),
    ( "\t\tTOPS=" +
      ( [.top_logprobs[]? | (.token | @json) + ":" + (.logprob | tostring)] | join(",") )
    )
  ] | join("  ")
'