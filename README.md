# Maison Mode — Fashion Insights

**Maison Mode** is a fashion recommendation web app built with Python, Streamlit, Pandas, NumPy, and Scikit-learn. It explores consumer preferences and shopping behaviour using a fashion dataset.

## Features

- Filter fashion items by clothing type and occasion.
- Set a maximum budget.
- Adjust style preferences for trendiness, comfort, quality, and fit.
- Rank matching items using a style-match score.
- Estimate purchase likelihood with a machine-learning classifier.
- Predict a satisfaction score with a regression model.
- Explore basic dataset summaries and product-price insights.

## Project files

- `app.py` — Streamlit application.
- `fashion_dataset_complete.csv` — dataset used by the application.
- `requirements.txt` — Python dependencies.

## Run locally

1. Install Python 3.10 or newer.
2. Download or clone this repository.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure `fashion_dataset_complete.csv` is in the location expected by `app.py`.
5. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `fashion_dataset_complete.csv` to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect GitHub and choose the repository.
4. Set the main file path to `app.py` and deploy.

## Notes

The app's predictions are estimates based on the available dataset and trained models. They are not guarantees of actual customer behaviour. Make sure the dataset contains the columns expected by `app.py` before deployment.
