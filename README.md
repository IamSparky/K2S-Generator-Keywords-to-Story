# 🧠📝 Keyword2Story: TinyStories Generation with PyTorch + FLAN-T5

**Keyword2Story** turns just 5 keywords into rich, imaginative short stories using a fine-tuned `google/flan-t5-base` model.

This project is built to run efficiently on minimal GPUs, with clean keyword extraction, structured training, and evaluation pipelines.

---

## 🔗 Model Source

We fine-tune the [**google/flan-t5-base**](https://huggingface.co/google/flan-t5-base) model from Hugging Face, a powerful text-to-text transformer pre-trained on a wide range of instruction-following tasks.

---

## 📁 Project Structure

```
Keyword2Story/
│
├── data/
│   └── archived data file # Original dataset (downloaded from Kaggle)
│
├── outputs/
│   ├── stories_with_keywords_train.pkl # Keyword extracted Data using extractKeywords.py
│   └── stories_with_keywords_valid.pkl # Keyword extracted Data using extractKeywords.py
│
├── scripts/
│   └── keywordsExtractionScript/
|   |   └── dataCheck.ipynb
|   |   └── extractKeywords.py
|   |
|   └── storyGenerationScript/
|       └── jupyter notebook/
|       |   └── createStoryGenerationModel
|       |   
|       └── VS Code/
|           └── dataframeWithCrossValidation.py
|           └── dataset.py
|           └── trainer.py
|           └── evaluator.py
|           └── run.py 
│
└── README.md # This file
```


---

## 📦 1. Download the Dataset

We use the TinyStories dataset from Kaggle, which contains hundreds of small structured stories perfect for generation tasks.

📥 Download it from here:  
[**TinyStories - Narrative Classification**](https://www.kaggle.com/datasets/thedevastator/tinystories-narrative-classification)

Place the downloaded `tinystories.csv` inside the `data/` folder.

---

## 🔍 2. Extract Keywords

Using basic NLP filtering + TF-IDF, extract 5 keywords for each story:

You can use:
- `KeywordsExtractionScript/keyword_extraction_vs_code.py` for script-based usage
- `KeywordsExtractionScript/keyword_extraction_notebook.ipynb` for notebook-based processing

✅ Output: A new file `stories_with_keywords.pkl` in `data/` containing stories + keywords.

---

## 🧠 3. Train the Story Generation Model

Train a fine-tuned version of `flan-t5-base` to generate stories from keywords.

Training scripts:
- `StoryGenerationScript/story_generation_vs_code.py` — CLI version
- `StoryGenerationScript/story_generation_notebook.ipynb` — Jupyter Notebook

It uses:
- 5-fold cross-validation
- BLEU and ROUGE-L scores
- GPU-friendly setup with TQDM progress tracking

---

## 📈 Example Inference

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy = False)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto").to(DEVICE)
model.load_state_dict(torch.load("models/flan_t5_storygen_fold.pt"))

model.eval()
input_text = "generate story from keywords: robot, teddy, fix, help, steel"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

output = model.generate(input_ids, max_length=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 💡 Sample Output

**Input Keywords:**  
`robot, teddy, fix, help, steel`

**Generated Story:**  
> Once upon a time, there was a robot who saw a little girl crying because her teddy bear was torn.  
> The robot tore a piece of steel from his own body to help her fix it.  
> They became best friends and played in the park every day.

---

## 🧰 Technologies Used

- 🤗 **Hugging Face Transformers** (`flan-t5-base`)  
- 🔥 **PyTorch**  
- 📊 **NLTK**, **ROUGE**, **TQDM**  
- 🧪 **BLEU and ROUGE-L Metrics**  
- 💾 **Minimal GPU (6GB) supported**

---

## 👨‍💻 Created by Soumo

Feel free to explore, contribute, or remix this storytelling machine!  
Questions or improvements? **Open an issue** or **drop a ⭐ if you find it helpful!**
