$ Install any new packages: ```pip freeze > requirements.txt ```

## Instructions to run

```shell script
$ pip install -r requirements.txt
$ learningEnv/bin/activate 
OR (Mac)
$ learningEnv/bin/activate
```

### Training
```shell script
$ python train.py --help
Usage: train.py [OPTIONS]

Options:
  --num-episodes INTEGER  Number of episodes to train on  [default: 100000]
  --save-path TEXT        Path to save the Q-table dump  [default:
                          q_table.pickle]
  --help                  Show this message and exit.
```

### Evaluation

```shell script
$ python evaluate.py --help
Usage: evaluate.py [OPTIONS]

Options:
  --num-episodes INTEGER  Number of episodes to train on  [default: 100]
  --q-path TEXT           Path to read the q-table values from  [default:
                          q_table.pickle]
  --help                  Show this message and exit.
```