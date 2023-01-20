## Mitigating Side Impact of Over-association in Conversation Query Generation

This repository includes codes for the paper ''Mitigating the Negative Impact of Over-association for Conversational Query Production".

#### Base / Data-based
* Data preparation: Download datasets (Wizard-of-Internet or DuSinc) and place data file under "saved_data" directory. Run scripts under ''databased/prepare_data_xxx.py'' to process data.
* Training : Run scripts are under ''databased'' like "train_xx.sh". The models are also saved in "saved_data" directory. Model predictions are generated when training is done automatically, which is saved as "generated_predictions.txt" by default.
* Evaluation: Use "eval_xxx.py" under ''databased'' to evaluate. Please note the file path.

#### Model-based / Combine
* Data preparation: Use the trained models above to prepare generated candidates for model-wholeseq. For Dusinc, we recommand to train k model separately because of its small size (split the data set to k fold, and get pseudo data for i-th fold using data from other folds). In this way, we can get better candidates according with model distribution.
* Training: Run "modelbased/train.sh". ''en'' or ''zh'' for the second argument. The hyperparameters are in "config_xx.py" and "config_xx_zh.py" respectively.
* Predicting: Run "modelbased/test.sh" to generate predictions.
* Evaluation: Same as "Base / Data-based". Note the file path.
