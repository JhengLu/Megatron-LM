pip install datasets
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
pip install einops
pip install pybind11

unset PIP_CONSTRAINT      
rm -rf ~/venv‑clean                # be sure you target the right path!
python3 -m venv ~/venv-clean

source ~/venv-clean/bin/activate      # if not already active
pip install "dill==0.3.8"
pip install "datasets==3.6.0"         # or 2.21.0 if you prefer
python -c "
from datasets import load_dataset
import json

ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
with open('my-corpus.json', 'w', encoding='utf-8') as f:
    for item in ds:
        text = item['text'].strip()
        if text:
            f.write(json.dumps({'text': text}) + '\n')
"
deactivate

cd ..
python tools/preprocess_data.py \
  --input data/my-corpus.json \
  --output-prefix my-gpt2 \
  --vocab-file data/gpt2-vocab.json \
  --tokenizer-type GPT2BPETokenizer \
  --merge-file data/gpt2-merges.txt \
  --append-eod \
  --workers 4
