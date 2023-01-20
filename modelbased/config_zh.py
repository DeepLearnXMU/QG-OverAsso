data_dir = '../saved_data/data_dusinc'

class Arguments:
    train_file = f'{data_dir}/train_beam_10.json'
    valid_file = f'{data_dir}/dev.json'
    test_file = f'{data_dir}/dev.json'
    do_train = True
    do_eval = True
    do_predict = True
    model_name_or_path = '../saved_data/mengzi-t5-base-dusinc'
    max_source_length = 512
    max_target_length = 64
    learning_rate = 3e-5
    epoches = 5
    batch_size = 1
    eval_batch_size = 16
    gradient_accumulation_steps = 64
    report_steps = 50
    eval_steps = 50
    max_length = 64
    num_beams = 4
    norm = False
    norm_lambda = 1e-5
    topk = 10
    degree_threshold = None
    output_dir = '../saved_data/RAQG_zh_top10'
    predictions = '../saved_data/RAQG_zh_top10/generated_predictions.txt'