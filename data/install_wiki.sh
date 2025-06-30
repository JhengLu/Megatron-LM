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
