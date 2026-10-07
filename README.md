# olx-category-classifier

A small AI model that takes an offer title and returns suggested categories along with a confidence score (in percent).

```
Input:   "Sprzedam rower górski 26 cali"
Output:  Sport i Hobby » Rowery        87%
         Sport i Hobby » Fitness       4%
         Dla Dzieci » Rowerki          3%
```

> Illustrative example. Real results will be added after training.

A learning project covering the full path: from a simple baseline to fine-tuning a transformer model, probability calibration and deployment.

## Data

[Polish OLX items](https://www.kaggle.com/datasets/heolin/polish-olx-items) (Kaggle). Only two columns are used:

- `name`: offer title,
- `tag`: category in the format `category » subcategory`.

The data is not part of this repository. Download it from Kaggle and extract it into `data/`.

## Approach

1. **Data preparation**: cleaning, removing duplicates and label conflicts, filtering rare classes, a fixed train/val/test split.
2. **Baseline**: TF-IDF (character n-grams) + ~~logistic regression~~ SGDC Classifier.
3. **Fine-tuning**: HerBERT (`allegro/herbert-base-cased`) with Hugging Face Transformers.
4. **Calibration**: temperature scaling, so that the percentages actually reflect the model's confidence.
5. **Deployment**: export to ONNX and use from C# via ONNX Runtime (or a simple FastAPI service).

## Results

| Model                    | Accuracy | Top-3 accuracy | Macro F1 |
|--------------------------|----------|----------------|----------|
| TF-IDF + LogReg          | TBD      | TBD            | TBD      |
| TF-IDF + SGDClassifier   | 0.727461420449418  | 0.8695514845230575  | 0.601030076913735  |
| HerBERT (fine-tuned)     | TBD      | TBD            | TBD      |
| HerBERT + calibration    | TBD      | TBD            | TBD      |

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order:

| Notebook                    | Contents                                   |
|-----------------------------|--------------------------------------------|
| `01_data_preparation.ipynb` | data cleaning, train/val/test split        |
| `02_baseline.ipynb`         | TF-IDF + logistic regression               |
| `03_finetuning.ipynb`       | HerBERT fine-tuning                        |
| `04_calibration.ipynb`      | calibration and confidence evaluation      |

## Repository structure

```
├── data/           # dataset (git-ignored)
├── notebooks/
├── src/
├── requirements.txt
└── README.md
```

## Limitations

- The model is trained on short offer titles, so it may perform worse on long descriptions.
- Labels in the dataset can be noisy (the same title appearing in different categories).
- Rare categories were filtered out, so the model does not predict them.

## License

Code: MIT. The data is subject to the license listed on the dataset's Kaggle page.
