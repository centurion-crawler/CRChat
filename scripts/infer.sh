CUDA_VISIBLE_DEVICES=1 python -u llava/eval/model_vqa.py \
  --model-path ./checkpoints/<checkpoint_path> \
  --model-base ./Qwen \
  --conv-mode qwen_2 \
  --question-file ./playground/VQA_examples/questions.jsonl \
  --answers-file ./playground/VQA_examples/answers.jsonl