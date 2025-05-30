
# Source Code

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Training

To train the models used in the model-based baselines, run:

```bash
python train.py --dataset routerbench
```

## Running

To reproduce the main results for on RouterBench, run:

```bash
python test.py --ops 1 2 3 4 5 6 7 8 --N=10000 --alpha=0.0001 --eps=0.025 --budget=1 --split=weighted --embed=bge
```


