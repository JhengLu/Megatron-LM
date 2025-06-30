cd ..
python tools/preprocess_data.py \
  --input data/my-corpus.json \
  --output-prefix my-gpt2 \
  --vocab-file data/gpt2-vocab.json \
  --tokenizer-type GPT2BPETokenizer \
  --merge-file data/gpt2-merges.txt \
  --append-eod \
  --workers 4
