import torch
import gc

from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm

def eval_loop_fn(data_loader, model, tokenizer, device):
    model.eval()
    running_loss = 0.0

    all_predictions = []
    all_targets = []

    tqdm_ob = tqdm(data_loader, total=len(data_loader), desc="Evaluating")

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm_ob):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            running_loss += loss.item()

            preds = model.generate(input_ids=input_ids, max_length=256)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_predictions.extend(decoded_preds)
            all_targets.extend(decoded_labels)

            del input_ids, attention_mask, labels
            gc.collect()
            torch.cuda.empty_cache()

    # Average loss
    val_loss = running_loss / len(data_loader)

    # BLEU
    references = [[target.split()] for target in all_targets]
    candidates = [pred.split() for pred in all_predictions]
    bleu_score = corpus_bleu(references, candidates)

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(all_targets, all_predictions)]
    rouge_l_score = sum(rouge_scores) / len(rouge_scores)

    return val_loss, bleu_score, rouge_l_score
