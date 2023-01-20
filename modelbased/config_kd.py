data_dir = '../saved_data/data_woi'

class Arguments:
    train_file = f'{data_dir}/train.json'
    valid_file = f'{data_dir}/valid.json'
    test_file = f'{data_dir}/test.json'
    do_train = True
    do_eval = True
    do_predict = True
    model_name_or_path = '../saved_data/t5-v1_1-base-woi_exponent-2'
    max_source_length = 1024
    max_target_length = 64
    learning_rate = 1e-5
    epoches = 10
    batch_size = 8
    eval_batch_size = 16
    gradient_accumulation_steps = 32
    report_steps = 50
    eval_steps = 50
    max_length = 64
    num_beams = 4
    alpha = 0.75
    output_dir = '../saved_data/RAQG_exponent-2_KD_0.75'
    predictions = '../saved_data/RAQG_exponent-2_KD_0.75/generated_predictions.txt'
