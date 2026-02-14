#!/bin/bash

# jsonの配列を作成
JSON_EMB="["

for i in {1..10000} ; do
    JSON_EMB+='{"key":"test_'${i}'","path":"models/jp_male.emb.gguf"},'
done
JSON_EMB+='{"key":"en_female","path":"models/en_female.emb.gguf"},'
JSON_EMB+='{"key":"en_male","path":"models/en_male.emb.gguf"},'
JSON_EMB+='{"key":"jp_female","path":"models/jp_female.emb.gguf"},'
JSON_EMB+='{"key":"jp_male","path":"models/jp_male.emb.gguf"}'

JSON_EMB+="]"
# echo $JSON_EMB

./build/mio-tts-server \
    -m models/MioTTS-0.1B-Q8_0.gguf \
    -mv models/miocodec.gguf \
    --tts-wavlm-model models/wavlm_base_plus_2l_f32.gguf \
    -ngl 99 \
    --ctx-size 400 \
    --llm-shared-context off \
    --parallel 1 \
    --parallel-reference-generation 1 \
    --flash-attn on \
    --reference-file-json '[
{"key":"en_female","path":"models/en_female.emb.gguf"},
{"key":"en_male","path":"models/en_male.emb.gguf"},
{"key":"jp_female","path":"models/jp_female.emb.gguf"},
{"key":"jp_male","path":"models/jp_male.emb.gguf"}
]' \
    # --port 18089 \
    # --llm-api-url http://127.0.0.1:8080/v1/chat/completions \
    # --llm-api-key "dummy" \

    # --llm-api-headers '{"CF-Access-Client-Id": "<CF_ACCESS_CLIENT_ID>", "CF-Access-Client-Secret": "<CF_ACCESS_CLIENT_SECRET>"}' \
    # --llm-api-url http://127.0.0.1:8081/v1/chat/completions \
    # --llm-api-key "dummy" \
    # --flash-attn on \
    # --threads 16 \
    # --llm-api-url http://127.0.0.1:8080/v1/chat/completions \
    # --llm-api-key "dummy" \
    # --llm-api-model "MioTTS-0.1B-Q8_0.gguf" \
