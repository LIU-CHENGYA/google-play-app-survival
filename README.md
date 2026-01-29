# Google Play App Survival Analysis using NLP and Machine Learning

This project analyzes language-learning apps on the Google Play Store to investigate whether textual and semantic features extracted from app descriptions and user reviews can predict an app’s survival status or popularity.

The project integrates Natural Language Processing (NLP), machine learning, and a research-oriented experimental pipeline to examine how textual signals relate to app longevity.

The study is inspired by Havakhor et al. (2023), which demonstrates that early textual narratives on social media can serve as strong signals for predicting startup fundraising success. Similarly, this project explores whether linguistic signals embedded in app-related texts convey meaningful information about app survival.


---

##  Research Questions (RQ)

### RQ1：Can semantic and linguistic features from app descriptions predict whether an app survives?

* Extract linguistic and semantic features from app descriptions and user reviews

* Train a Logistic Regression model

* Evaluation metrics: Accuracy, ROC-AUC

---

### RQ2：Does semantic differentiation from competitors affect app survival?

* Use K-Means clustering to group apps into semantically similar clusters (treated as competitive groups)

* Define semantic differentiation as:

    * The distance between an app and its cluster centroid (mean feature values)

* Introduce this distance as a new feature and retrain the survival prediction model

---

### RQ3：Is semantic consistency between app descriptions and user reviews associated with app survival?

* Compute cosine similarity between description and review embeddings

* Analyze whether higher semantic alignment predicts survival

---

### RQ4：Which linguistic features are most predictive of app survival?

* Train a model using multiple linguistic features simultaneously

* Analyze feature importance via Logistic Regression coefficients

---
## Survival Definition
* Survived (1): App updated within the last 90 days

* Not survived (0): App not updated for more than 90 days


##  Features Used

* Text length (description / review)
* Unique word count
* Lexical diversity
* Average word length
* Description–Review cosine similarity
* Semantic distance to cluster center (RQ2)

All features are standardized to ensure model stability and meaningful distance calculations.

---

##  Project Structure

```
google-play-app-survival/
│
├── data/
│ ├── raw/ # 原始爬取或下載資料
│ ├── processed/ # 清洗後資料
│ └── features/ # 特徵工程與模型輸入資料
|
├── result_data/   
│   ├── coef_df_rq*.csv
|   ├── evaluate_result_rq*.json
|
├── src/
│   ├── __init__.py
|   ├── file_utils.py        # 載入和儲存檔案
│   ├── scraping.py          # 資料蒐集（或 mock）
│   ├── preprocessing.py     # 清洗、斷詞、正規化
│   ├── features.py          # TF-IDF、文本指標
│   ├── clustering.py        # k-means + 距離計算
│   ├── labels.py            # 存活標籤
│   ├── model.py             # 預測模型
│   ├── experiment.py
│
├── notebooks/
│   └── exploration.ipynb    # 嘗試模型
│
├── main.py                  # 跑完整 pipeline
├── README.md
```

---

##  How to Run

```bash
python main.py
```

Running the script will:

* Execute experiments for RQ1–RQ4

* Save evaluation results as JSON files

* Save feature importance results as CSV files

---

##  Models & Evaluation

* Model: Logistic Regression
* Metrics:

  * Accuracy
  * ROC-AUC
  * Classification Report

部分實驗中，因類別不平衡，可能出現 `UndefinedMetricWarning`，此為實務與研究中常見現象，並不影響 ROC-AUC 作為主要比較指標。

---

##  Key Takeaways

* Linguistic and semantic features can meaningfully predict app survival to a certain extent.
* The completeness and depth of textual content are more explanatory than semantic differentiation alone.
* Semantic distance from competitors and description–review consistency exhibit limited impact on survival.

---

##  Skills Demonstrated

* Python (pandas, numpy, scikit-learn)
* NLP feature engineering
* Unsupervised learning (K-Means)
* Supervised learning & evaluation
* Experiment pipeline design
* Research-oriented thinking & interpretation

---
## Reference

Havakhor, T., Golmohammadi, A., Sabherwal, R., & Gauri, D. (2023). Do Early Words from New Ventures Predict Fundraising? A Comparative View of Social Media Narratives. MIS Quarterly, 47(2), 611–638. https://doi.org/10.25300/misq/2022/16392

##  License

This project is for academic and portfolio demonstration purposes.
