# Dataset

The notebook expects a Sentiment140-style CSV with binary positive/negative labels and tweet text. The saved run used 1,652,310 rows after loading and cleaning.

The dataset is not included in this repository. Download it from its authorized source, review its license, and save a local copy as:

```text
data/sentiment140.csv
```

The notebook detects common column names such as `target`, `sentiment`, `polarity`, `label`, and `class`. It maps the original binary labels to `0` (negative) and `1` (positive), removes unusable rows, and creates a seeded train/test split.

Do not commit raw tweets, trained checkpoints, or cached Hugging Face datasets. They can be large and may have separate usage or redistribution conditions.

