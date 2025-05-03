import torch
import gc

from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.amp import GradScaler


def train_loop_fn(data_loader, model, tokenizer, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0

    all_predictions = []
    all_targets = []

    scaler = GradScaler(device='cuda')

    tqdm_ob = tqdm(data_loader, total=len(data_loader), desc="Training")

    for batch_index, batch in enumerate(tqdm_ob):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

        # Decode predictions and labels
        preds = model.generate(input_ids=input_ids, max_length=256)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        all_predictions.extend(decoded_preds)
        all_targets.extend(decoded_labels)

        del input_ids, attention_mask, labels
        gc.collect()
        torch.cuda.empty_cache()

    # Compute average loss
    train_loss = running_loss / len(data_loader)

    # Compute BLEU score
    references = [[target.split()] for target in all_targets]  # list of list of list of words
    candidates = [pred.split() for pred in all_predictions]    # list of list of words
    bleu_score = corpus_bleu(references, candidates)

    # Compute ROUGE-L score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(all_targets, all_predictions)]
    rouge_l_score = sum(rouge_scores) / len(rouge_scores)

    return train_loss, bleu_score, rouge_l_score
