data_dir = '../saved_data/data_dusinc'

class Arguments:
    train_file = f'{data_dir}/train.json'
    valid_file = f'{data_dir}/dev.json'
    test_file = f'{data_dir}/dev.json'
    do_train = True
    do_eval = True
    do_predict = True
    model_name_or_path = '../saved_data/mengzi-t5-base-dusinc_exponent-0.5'
    max_source_length = 1024
    max_target_length = 64
    learning_rate = 1e-5
    epoches = 5
    batch_size = 8
    eval_batch_size = 16
    gradient_accumulation_steps = 8
    report_steps = 50
    eval_steps = 50
    max_length = 64
    num_beams = 4
    alpha = 0.75
    output_dir = '../saved_data/RAQG_exponent-0.5_KD_zh_0.75'
    predictions = '../saved_data/RAQG_exponent-0.5_KD_zh_0.75/generated_predictions.txt'
