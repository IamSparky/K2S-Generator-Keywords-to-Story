from torch.utils.data import Dataset
from transformers import T5Tokenizer

# Define the Dataset
class KeywordsToStoryDataset(Dataset):
    def __init__(self, dataframe, tokenizer_name="google/flan-t5-base", max_input_length=32, max_target_length=128):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.data = dataframe
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        keywords = ", ".join([str(row[f"Keyword {i}"]) for i in range(1, 6)])
        input_text = f"generate story from keywords: {keywords}"
        target_text = str(row["text"])

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze(),
            "input_text": input_text,
            "target_text": target_text
        }