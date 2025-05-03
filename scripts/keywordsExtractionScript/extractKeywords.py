# Re-import everything after reset to ensure all necessary packages are loaded
import pandas as pd
import re
from collections import Counter
from tqdm import tqdm

# Redefine file paths
csv_file_path = "../../data/train.csv"
pkl_file_path = "../../outputs/stories_with_keywords_train.pkl"

# Load the CSV file
df = pd.read_csv(csv_file_path, low_memory=False)

# Initialize tqdm for pandas apply
tqdm.pandas()

# Define keyword extraction function with tqdm progress
def extract_keywords_list(text, top_n=5):
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn", "doesn", "don't", "doesn't",
    "didn't", "won't", "wouldn't", "shouldn't", "can't", "couldn't", "mustn't", "needn't",
    "mightn't", "shan't", "n't"
])
    filtered_words = [word for word in words if word not in stopwords]
    most_common = Counter(filtered_words).most_common(top_n)
    return [word for word, _ in most_common]

# Apply keyword extraction with progress bar
keywords = df.iloc[:, 0].astype(str).progress_apply(extract_keywords_list)

# Expand keyword lists into separate columns
df[['Keyword 1', 'Keyword 2', 'Keyword 3', 'Keyword 4', 'Keyword 5']] = pd.DataFrame(keywords.tolist(), index=df.index)

# Save the updated DataFrame as a .pkl file
df.to_pickle(pkl_file_path)

