import sys
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from dataset import RAQGDataset, QueryGenDataset
from utils import *
from config_ra import *
from model_ra import T5ForConditionalGenerationRAQG
from config_zh import Arguments as Arguments_zh

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGenerationRAQG.from_pretrained(args.model_name_or_path)
    model = model.cuda()

    def collate_fn(batch):
        inputs, targets, candidates, scores = zip(*batch)
        inputs, targets, candidates, scores = flat_list(inputs), flat_list(targets), \
                                              flat_list(candidates), flat_list(scores)
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=True,
                                 truncation=True, return_tensors='pt')

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(candidates, max_length=args.max_target_length, padding=True,
                               truncation=True, return_tensors='pt')
            labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels["input_ids"]

        # prepare decoder_input_ids
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
        model_inputs["decoder_input_ids"] = decoder_input_ids
        model_inputs["rewards"] = torch.tensor(scores)

        return model_inputs

    def collate_fn_for_test(batch):
        inputs, targets = zip(*batch)
        model_inputs = tokenizer(list(inputs), max_length=args.max_source_length, padding=True,
                                 truncation=True, return_tensors='pt')

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(list(targets), max_length=args.max_target_length, padding=True,
                               truncation=True, return_tensors='pt')
            labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels["input_ids"]

        # prepare decoder_input_ids
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
        model_inputs["decoder_input_ids"] = decoder_input_ids

        return model_inputs

    if args.do_train:
        train_dataset = RAQGDataset(args.train_file, args.topk, args.degree_threshold)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    if args.do_eval:
        valid_dataset = QueryGenDataset(args.valid_file, args.degree_threshold)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn_for_test)
    if args.do_predict:
        test_dataset = QueryGenDataset(args.test_file, args.degree_threshold)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn_for_test)

    # if args.do_eval:
    #     eval_loss = 0.
    #     eval_steps = 0
    #     generated_predictions = []
    #     gold_references = []
    #     model.eval()
    #     with torch.no_grad():
    #         for eval_batch in tqdm(valid_dataloader):
    #             eval_steps += 1
    #             eval_batch = preprocess_batch(eval_batch)
    #             eval_loss += model(**eval_batch).loss.item()
    #             if 'decoder_input_ids' in eval_batch:
    #                 eval_batch.pop('decoder_input_ids')
    #             if 'labels' in eval_batch:
    #                 gold_references += tokenizer.batch_decode(fix_labels(eval_batch.pop('labels')),
    #                                                           skip_special_tokens=True,
    #                                                           clean_up_tokenization_spaces=True)
    #             predictions = model.generate(**eval_batch, max_length=args.max_length, num_beams=args.num_beams)
    #             generated_predictions += tokenizer.batch_decode(predictions, skip_special_tokens=True,
    #                                                             clean_up_tokenization_spaces=True)
    #         eval_loss = round(eval_loss / eval_steps, 4)
    #     best_loss = eval_loss
    #     print(f'evaluate steps: 0, loss: {eval_loss}, best loss: {best_loss}, '
    #           f'uni. F1: {uni_F1_score(generated_predictions, gold_references)}')

    if args.norm:
        raw_params = torch.clone(torch.cat([x.view(-1).detach() for x in model.parameters()]))

    if args.do_train:
        backward_steps, global_steps = 0, 0
        train_loss, best_loss = 0., 1e9
        model.train()
        for epoch in range(args.epoches):
            for batch in train_dataloader:
                try:
                    batch = preprocess_batch(batch)
                    loss = model.forward_ra(**batch) / args.gradient_accumulation_steps
                    train_loss += loss.item()
                    if args.norm:
                        loss += torch.sum(torch.abs(torch.cat([x.view(-1) for x in model.parameters()]) - raw_params)) \
                                * args.norm_lambda / args.gradient_accumulation_steps
                    loss.backward()
                    backward_steps += 1
                except RuntimeError:
                    print(f'OOM (step {global_steps}): check if this frequently occurs!')
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    backward_steps = 0 # recount
                    continue
                if backward_steps % args.gradient_accumulation_steps == 0:
                    global_steps += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    if global_steps % args.report_steps == 0:
                        print(f'training epoches: {epoch}, steps: {global_steps}, loss: {round(train_loss/global_steps, 4)}')
                    if args.do_eval and global_steps % args.eval_steps == 0:
                        eval_loss = 0.
                        eval_steps = 0
                        generated_predictions = []
                        gold_references = []
                        model.eval()
                        with torch.no_grad():
                            for eval_batch in valid_dataloader:
                                eval_steps += 1
                                eval_batch = preprocess_batch(eval_batch)
                                eval_loss += model(**eval_batch).loss.item()
                                if 'decoder_input_ids' in eval_batch:
                                    eval_batch.pop('decoder_input_ids')
                                if 'labels' in eval_batch:
                                    gold_references += tokenizer.batch_decode(fix_labels(eval_batch.pop('labels')),
                                                                              skip_special_tokens=True,
                                                                              clean_up_tokenization_spaces=True)
                                predictions = model.generate(**eval_batch, max_length=args.max_length,
                                                             num_beams=args.num_beams)
                                generated_predictions += tokenizer.batch_decode(predictions, skip_special_tokens=True,
                                                                                clean_up_tokenization_spaces=True)
                            eval_loss = round(eval_loss / eval_steps, 4)
                        best_loss = eval_loss
                        save_model(args, model, tokenizer, step=global_steps)
                        print(f'evaluate steps: {global_steps}, loss: {eval_loss}, best loss: {best_loss}, '
                              f'uni. F1: {uni_F1_score(generated_predictions, gold_references)}')
                        model.train()
        if args.do_eval:
            eval_loss = 0.
            eval_steps = 0
            generated_predictions = []
            gold_references = []
            model.eval()
            with torch.no_grad():
                for eval_batch in valid_dataloader:
                    eval_steps += 1
                    eval_batch = preprocess_batch(eval_batch)
                    eval_loss += model(**eval_batch).loss.item()
                    if 'decoder_input_ids' in eval_batch:
                        eval_batch.pop('decoder_input_ids')
                    if 'labels' in eval_batch:
                        gold_references += tokenizer.batch_decode(fix_labels(eval_batch.pop('labels')),
                                                                  skip_special_tokens=True,
                                                                  clean_up_tokenization_spaces=True)
                    predictions = model.generate(**eval_batch, max_length=args.max_length, num_beams=args.num_beams)
                    generated_predictions += tokenizer.batch_decode(predictions, skip_special_tokens=True,
                                                                    clean_up_tokenization_spaces=True)
                eval_loss = round(eval_loss / eval_steps, 4)
            best_loss = eval_loss
            save_model(args, model, tokenizer, step=global_steps)
            print(f'evaluate steps: {global_steps}, loss: {eval_loss}, best loss: {best_loss}, '
                  f'uni. F1: {uni_F1_score(generated_predictions, gold_references)}')
    if args.do_predict:
        eval_loss = 0.
        eval_steps = 0
        generated_predictions = []
        gold_references = []
        model.eval()
        with torch.no_grad():
            for eval_batch in tqdm(test_dataloader):
                eval_steps += 1
                eval_batch = preprocess_batch(eval_batch)
                eval_loss += model(**eval_batch).loss.item()
                if 'decoder_input_ids' in eval_batch:
                    eval_batch.pop('decoder_input_ids')
                if 'labels' in eval_batch:
                    gold_references += tokenizer.batch_decode(fix_labels(eval_batch.pop('labels')),
                                                              skip_special_tokens=True,
                                                              clean_up_tokenization_spaces=True)
                predictions = model.generate(**eval_batch, max_length=args.max_length, num_beams=args.num_beams)
                generated_predictions += tokenizer.batch_decode(predictions, skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=True)
            eval_loss = round(eval_loss / eval_steps, 4)
        print(f'test loss: {eval_loss} uni. F1: {uni_F1_score(generated_predictions, gold_references)}')
        with open(args.predictions, 'w') as f:
            f.write('\n'.join(generated_predictions))

if __name__ == '__main__':
    if sys.argv[1] == 'en':
        args = Arguments()
    elif sys.argv[1] == 'zh':
        args = Arguments_zh()
    else:
        raise Exception
    main(args)
