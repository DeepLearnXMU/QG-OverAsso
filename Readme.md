## Multi-view Query Generation

This repository includes codes for the paper ''Multi-View Conversational Query Production".

#### Base / Static-view
* Data preparation: Download datasets (Wizard-of-Internet or DuSinc) and place data file under "saved_data" directory. Run scripts under ''data'' to process data.
* Training : All scripts are under ''script'' like "train_xx.sh". The models are also saved in "saved_data" directory.
* Predicting: All models can use the same script ''script/test.sh'' to generate predictions for the final model (e.g. "t5-v1_1-base) or a checkpoint (e.g. "t5-v1_1-base/checkpoint-5000"). The output file name is "generated_predictions.txt" by default.
* Evaluation: Use eval.py under ''data'' to evaluate. Please note the file path.

#### Dynamic-view / Multi-view
* Data preparation: Use the trained models above to prepare generated candidates. We recommand to train k model separately (split the data set to k fold, and get pseudo data for i-th fold using data from other folds). In this way, we can get better candidates according with model distribution. The k-fold data has been processed in the above steps. You can find them in "saved_data" repository.
* Training: Run train.sh. ''en'' or ''zh'' for the second argument. The hyperparameters are in "config_ra.py" and "config_zh.py" respectively.
* Predicting: Run test.sh to generate predictions.
* Evaluation: Same as "Base / Static-view". Note the file path.
