data_dir = '../saved_data/data_woi'

class Arguments:
    train_file = f'{data_dir}/train_beam_10.json'
    valid_file = f'{data_dir}/valid.json'
    test_file = f'{data_dir}/test.json'
    do_train = True
    do_eval = True
    do_predict = True
    model_name_or_path = '../saved_data/t5-v1_1-base-woi'
    max_source_length = 1024
    max_target_length = 64
    learning_rate = 3e-5
    epoches = 20
    batch_size = 1
    eval_batch_size = 16
    gradient_accumulation_steps = 256
    report_steps = 50
    eval_steps = 50
    max_length = 64
    num_beams = 4
    norm = False
    norm_lambda = 1e-5
    topk = 10
    degree_threshold = None
    output_dir = '../saved_data/RAQG_top10'
    predictions = '../saved_data/RAQG_top10/generated_predictions.txt'
