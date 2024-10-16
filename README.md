# FOSS

A Pytorch Implementation of FOSS: A Self-Learned Doctor for Query Optimizers (ICDE 2024), Updated Version.

### Requirments

- Python 3.7
- Pytorch 1.12
- Ray RLlib 2.4.0

### PostgreSQL

PostgreSQL: v12.1

pg_hint_plan: We adopt the version specified in HybridQO. Install it following the document [https://github.com/yxfish13/PostgreSQL12.1_hint].

### Running

1. Modify config.py according to the PostgreSQL settings and setting the experiment in config.py or input the suitable parameters
2. run

```sh
    python ./FOSS/run_parallel.py
```

### Result

```sh
    tensorboard --logdir './runstate'
```

OR

```
  Examine the JSON document located in the 'timely_result' folder.
```
