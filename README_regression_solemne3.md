# Predicting Review Ratings with Regression Models

Regression analysis for **Solemne 3** of the TDS course (Universidad San Sebastián). The notebook builds and compares several regression models that predict a customer's review rating from 17 descriptors of their purchases, selects the 10 most informative variables, and studies the effect of normalizing the data.

**Notebook:** `Solemne_3_TDS.ipynb`

<!-- Add your Colab badge here, e.g.:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK)
-->

## Objective

- Build and evaluate regression models that predict a satisfaction-type score from 17 automatically collected descriptors.
- Use at least 3 different algorithms.
- Select the 10 best variables with at least one method.
- Study the effect of normalizing the data.

## Data

File: `dataset3.csv` (3,900 rows, 18 columns, no missing values).

- **Target:** `Review Rating`. The dataset has no column called "satisfaction score", so the notebook uses this numeric rating as the satisfaction measure.
- **Numerical features (3):** `Age`, `Purchase Amount (USD)`, `Previous Purchases`.
- **Categorical features (14):** `Gender`, `Item Purchased`, `Location`, `Size`, `Color`, `Season`, `Subscription Status`, `Payment Method`, `Shipping Type`, `Shipping Type.1`, `Discount Applied`, `Promo Code Used`, `Preferred Payment Method`, `Frequency of Purchases`.

The data file is not included in this repository. Upload it to Colab or adjust the path at the top of the notebook.

## Approach

1. **Preprocessing pipeline**
   - Numerical features: median imputation (optionally followed by `StandardScaler`).
   - Categorical features: most-frequent imputation and one-hot encoding, which gives 150 features in total.
2. **Train/test split:** 80/20 (3,120 training rows and 780 test rows), with a fixed random seed.
3. **Models:** three families covering linear, bagging and boosting approaches.
   - Ridge regression
   - Random Forest regressor
   - Gradient Boosting regressor
4. **Hyperparameter search:** `RandomizedSearchCV` with 5-fold cross-validation on the training set, optimizing RMSE (30 iterations).
5. **Normalization study:** each model is trained with and without `StandardScaler` on the numerical features.
6. **Feature selection (top 10)**
   - Method 1: `SelectKBest` with `mutual_info_regression`.
   - Method 2: permutation importance on the best Random Forest.
   - All three models are re-trained using only the 10 selected variables.
7. **Metrics:** RMSE, MAE and R² on the test set.
8. **Saving:** the trained models and results are saved to a `models/` folder with `joblib`.

## Results

Test set performance:

| Model | Scaling | RMSE | MAE | R² |
|---|---|---|---|---|
| Random Forest | No | **0.7381** | **0.6459** | 0.0010 |
| Random Forest | Yes | 0.7381 | 0.6459 | 0.0010 |
| Gradient Boosting | No | 0.7387 | 0.6471 | -0.0008 |
| Gradient Boosting | Yes | 0.7387 | 0.6471 | -0.0008 |
| Ridge | No | 0.7437 | 0.6478 | -0.0145 |
| Ridge | Yes | 0.7437 | 0.6478 | -0.0144 |

**Best hyperparameters found:** Ridge `alpha=100`; Random Forest `n_estimators=400`, `max_depth=5`, `min_samples_split=10`; Gradient Boosting `n_estimators=100`, `max_depth=3`, `learning_rate=0.01`. These were the same with and without scaling.

### Using only the top 10 features

| Model | RMSE (all features) | RMSE (top 10) | R² (top 10) |
|---|---|---|---|
| Ridge | 0.7437 | 0.7385 | -0.0003 |
| Random Forest | 0.7381 | 0.7409 | -0.0067 |
| Gradient Boosting | 0.7387 | 0.7393 | -0.0024 |

Top 10 features by mutual information: `Item Purchased_Shirt`, `Location_Iowa`, `Location_North Dakota`, `Color_Indigo`, `Color_Lavender`, `Color_Orange`, `Color_Violet`, `Subscription Status_No`, `Payment Method_Credit Card`, `Payment Method_Debit Card`.

Most important features by permutation importance (Random Forest): `Purchase Amount (USD)`, `Age`, `Color_Gray`, `Payment Method_Debit Card`, `Season_Spring`, followed by several item and color categories.

### Takeaways

- **All models perform about the same as predicting the average rating.** Every R² is close to zero (between -0.015 and 0.001) and RMSE is about 0.74 for every model, so none of them explains much of the variation in `Review Rating`.
- **This suggests the 17 descriptors carry very little information about the rating.** Model choice and tuning make almost no difference.
- **Normalization had no meaningful effect.** RMSE changes by less than 0.00003 between the scaled and unscaled runs, and the best hyperparameters did not change. This is expected for tree-based models, and Ridge has only 3 numerical features to rescale.
- **Feature selection also changed little.** The two selection methods pick different variables (mutual information favors categories such as colors and locations, while permutation importance ranks purchase amount and age first), and importance values are very small in both cases.

## Tech stack

- Python 3
- pandas, NumPy
- scikit-learn (`Pipeline`, `ColumnTransformer`, `RandomizedSearchCV`, `Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`)
- matplotlib
- joblib

## How to run

1. Open `Solemne_3_TDS.ipynb` in Google Colab (or Jupyter).
2. Upload `dataset3.csv` next to the notebook (or edit `dataset_path`).
3. Run the cells in order. The hyperparameter searches are the slowest step.

## Limitations and possible improvements

- With R² near zero, these models should not be used for real predictions of review ratings.
- The rating is treated as a continuous variable, but it takes a limited range of values, so classification or ordinal approaches could be tried.
- Possible next steps: engineer new features (for example spend relative to the customer's history), try other models (SVR, XGBoost), and check whether the rating has any relationship with the features at all before tuning further.
