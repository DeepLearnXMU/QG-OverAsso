data_dir = '../saved_data/woi_data_np'


class Arguments:
    train_file = f'train.json'
    valid_file = f'{data_dir}/valid.json'
    test_file = f'{data_dir}/valid.json'
    do_train = True
    do_eval = True
    do_predict = True
    model_name_or_path = '../saved_data/t5-v1_1-base'
    max_source_length = 1024
    max_target_length = 64
    learning_rate = 3e-5
    epoches = 5
    batch_size = 1
    eval_batch_size = 16
    gradient_accumulation_steps = 256
    report_steps = 50
    eval_steps = 50
    max_length = 64
    num_beams = 4
    norm = False
    norm_lambda = 1e-5
    topk = -1
    none_adjust = 1.0
    output_dir = '../saved_data/RAQG'
    predictions = '../saved_data/RAQG/generated_predictions.txt'
