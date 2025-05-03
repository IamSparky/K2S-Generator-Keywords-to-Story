# 🧠📝 TinyStories: Keyword-to-Story Generation with PyTorch + FLAN-T5

Welcome to the **TinyStories Project**, where we turn a handful of keywords into complete short stories using the power of PyTorch and FLAN-T5!

This project follows a streamlined 3-step pipeline:

---

## 📦 1. Download the Dataset

We use the **TinyStories - Narrative Classification** dataset from Kaggle, which contains a wide range of small, structured, and themed stories perfect for text generation tasks.

📥 Get it from here:  
🔗 [https://www.kaggle.com/datasets/thedevastator/tinystories-narrative-classification](https://www.kaggle.com/datasets/thedevastator/tinystories-narrative-classification)

Once downloaded, place the dataset file (typically a `.csv`) in your local `data/` directory or wherever you plan to run your scripts from.

---

## 🔍 2. Extract Keywords from Stories

Next, we extract **5 high-quality keywords** from each story using lightweight NLP techniques like TF-IDF and token filtering.

📂 Scripts for this step are in the folder:
📁 KeywordsExtractionScript/
│
├── keyword_extraction_vs_code.py # Run this with a Python interpreter or VS Code
├── keyword_extraction_notebook.ipynb # Run this interactively in Jupyter Notebook


✅ Output: A new PKL file where each row contains the original story + 5 extracted keywords.

---

## 🧠 3. Train the PyTorch Story Generator

With the extracted keywords in hand, we fine-tune a `flan-t5-base` model to **generate short stories from 5 input keywords**.

📂 Training scripts are in:
📁 StoryGenerationScript/
│
├── story_generation_vs_code.py # Full training pipeline in Python (VS Code style)
├── story_generation_notebook.ipynb # Interactive notebook version (Jupyter-friendly)
├── trainer.py # Core training + evaluation loops

yaml
Copy
Edit

✨ Features:
- 5-Fold Cross Validation for robust evaluation
- BLEU and ROUGE-L metrics
- Progress tracked using `tqdm`
- Minimal memory usage (designed for 6GB GPUs)

---

## ⚙️ Tech Stack

- 🤖 `transformers` (Hugging Face)
- 🔥 `PyTorch` with AMP
- 📊 `nltk`, `rouge-score`, `tqdm`, `pandas`
- 💻 Trained on local GPU / Amazon SageMaker (optional deployment-ready)

---

## 🚀 Output Example

**Input Keywords:**
robot, teddy, fix, help, steel

markdown
Copy
Edit

**Generated Story:**
Once upon a time, there was a robot who saw a little girl crying because her teddy bear was torn. The robot tore a piece of steel from his own body to help her fix it. They became best friends and played in the park every day.

css
Copy
Edit

---

## 🧪 Try It Yourself

Once trained, you can test the model like this:

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
model.load_state_dict(torch.load("flan_t5_storygen_fold.pt"))

input_text = "generate story from keywords: robot, teddy, fix, help, steel"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

output = model.generate(input_ids, max_length=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

🙌 Credits
Created with ❤️ by Soumo
Powered by PyTorch, Hugging Face, and pure storytelling magic.


