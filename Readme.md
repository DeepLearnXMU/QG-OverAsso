## Multi-view Query Generation

#### base or explicit-view

* Data preparation: download dataset and run script under ''/data''.
* Training : scripts for our models are under ''/script''.
* Predicting: all models can use the same script to generate predictions ''/script/test.sh''.
* Evaluation: use eval.py under ''/data'' to evaluate.

#### Implicit-View (RAQG)

* Data preparation: use the trained models above to prepare generated candidates. You can use a single model instead of k models for k folds, we find the influence is limited.
* Training: run train.sh. ''en'' or ''zh'' for the second argument. note the path in config.py.
* Predicting: run test.sh.
* Evaluation: use eval.py or eval_zh.py to evaluate.

