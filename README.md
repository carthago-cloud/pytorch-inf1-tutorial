## Prep CPU/GPU model
```bash
$ python3.7 cpu_model.py
```

## Prep model for Inf1
```bash
$ python3.7 neuron_model.py
```

## Run API on CPU
```bash
$ uvicorn --host 0.0.0.0 cpu_api:app --reload
```

## Run API on Inf1
```bash
$ uvicorn --host 0.0.0.0 neuron_api:app --reload
```
