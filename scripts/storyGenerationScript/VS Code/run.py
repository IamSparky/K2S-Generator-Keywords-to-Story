import dataframeWithCrossValidation
from dataset import KeywordsToStoryDataset
from trainer import train_loop_fn
from evaluator import eval_loop_fn

from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

import torch

def run():
    # Constants
    EPOCHS = 2
    TRAIN_BATCH_SIZE = 1
    VALID_BATCH_SIZE = 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer & Model
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto").to(DEVICE)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2, T_mult=1, eta_min=1e-6, last_epoch=-1
    )

    for fold in range(1, 6):
        print("*" * 20, f"FOLD NUMBER {fold}", "*" * 20)

        df_train = dataframeWithCrossValidation.df[dataframeWithCrossValidation.df["Fold"] != fold].reset_index(drop=True)
        df_valid = dataframeWithCrossValidation.df[dataframeWithCrossValidation.df["Fold"] == fold].reset_index(drop=True)

        train_dataset = KeywordsToStoryDataset(df_train)
        valid_dataset = KeywordsToStoryDataset(df_valid)

        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
        val_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=2)

        for epoch in range(EPOCHS):
            print(f"Epoch --> {epoch + 1} / {EPOCHS}")
            print("-------------------------------")

            train_metrics = train_loop_fn(train_loader, model, tokenizer, optimizer, DEVICE, scheduler)
            print("Training Loss & Metrics:")
            print(f"Loss: {train_metrics[0]:.4f}, BLEU: {train_metrics[1]:.4f}, ROUGE-L: {train_metrics[2]:.4f}")

            val_metrics = eval_loop_fn(val_loader, model, tokenizer, DEVICE)
            print("Validation Loss & Metrics:")
            print(f"Loss: {val_metrics[0]:.4f}, BLEU: {val_metrics[1]:.4f}, ROUGE-L: {val_metrics[2]:.4f}")

        print("\n")

    # Save final model weights
    torch.save(model.state_dict(), '../../../outputs/flan_t5_storygen_5fold_model.pt')


if __name__ == "__main__":
    run()