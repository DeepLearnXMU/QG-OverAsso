import copy
import sys
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from dataset import QueryGenDataset
from utils import *
from config_kd import Arguments as Arguments_en
from config_kd_zh import Arguments as Arguments_zh
import torch.nn.functional as F

def reward_fn(reward_model, model_inputs, alpha=0.5):
    reward_model.eval()
    with torch.no_grad():
        outputs = reward_model(**model_inputs)

    logits = outputs.logits  # (B, L, V)
    logits = F.softmax(logits, dim=-1)

    labels = torch.clone(model_inputs["labels"])
    labels[labels < 0] = reward_model.config.pad_token_id
    labels = F.one_hot(labels, num_classes=logits.shape[-1])

    return alpha * logits + (1 - alpha) * labels

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model = model.cuda()

    reward_model = copy.deepcopy(model)
    reward_model.eval()

    def collate_fn(batch):
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
        train_dataset = QueryGenDataset(args.train_file)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-6)
    if args.do_eval:
        valid_dataset = QueryGenDataset(args.valid_file)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    if args.do_predict:
        test_dataset = QueryGenDataset(args.test_file)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    if args.do_train:
        backward_steps, global_steps = 0, 0
        train_loss, best_loss = 0., 1e9
        model.train()
        for epoch in range(args.epoches):
            for batch in tqdm(train_dataloader):
                batch = preprocess_batch(batch)
                outputs = model(**batch)
                logits = F.softmax(outputs.logits, dim=-1)
                logits = torch.clamp(logits, min=1e-8, max=1 - 1e-8)
                targets = reward_fn(reward_model, batch, args.alpha)
                attention_mask = (batch['labels'] != -100) & (batch['labels'] != tokenizer.pad_token_id)
                loss = torch.sum(- targets[attention_mask] * torch.log2(logits[attention_mask]), dim=-1).mean()
                loss = loss / args.gradient_accumulation_steps  # gradient accumulation
                train_loss += loss.item()
                loss.backward()
                backward_steps += 1

                if backward_steps % args.gradient_accumulation_steps == 0:
                    global_steps += 1
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        args = Arguments_en()
    elif sys.argv[1] == 'zh':
        args = Arguments_zh()
    else:
        raise Exception
    main(args)
