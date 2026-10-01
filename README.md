# Setup

ToolMol needs a Nautilus-hosted LLM for the agent and
the Boltz-2 service from https://github.com/Azhou2023/Boltz-API

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Keys

```bash
cp .env.example .env
```

- `NAUTILUS_KEY` — key for the hosted LLM at `https://ellm.nrp-nautilus.io/v1`. Keys
  are issued by NRP: https://nrp.ai/llmtoken/
- `BOLTZ_GATEWAY_URL` The URL that
  Boltz-API's `configure.sh` printed, e.g.
  `https://boltz-api-you.nrp-nautilus.io/predict_affinity`. 

Verify the LLM:

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
c = OpenAI(base_url='https://ellm.nrp-nautilus.io/v1', api_key=os.getenv('NAUTILUS_KEY'))
print([m.id for m in c.models.list().data])"
```

`gpt-oss` must appear in that list.

## 3. Boltz service

Deploy it per the README in
https://github.com/Azhou2023/Boltz-API and scale it to zero when you are done.

## 4. Run

E.g:
```bash
python run.py --max_obj qed --min_obj acaa1 sa --seed 0 \
  --oracle boltz --max_oracle_calls 1000 --run_name my_run
```

Targets: `c-met`, `brd4`, `acaa1`. Results land in
`results/<run_name>/seed_<n>.yaml` as `[affinity, call_index, qed, sa]`.

Seeds 0-4 on `c-met`, `brd4` and `acaa1` load a pre-scored starting population from
`boltz_caches/`, so the first 60 molecules cost no Boltz calls.
