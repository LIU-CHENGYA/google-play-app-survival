# Google Play App Survival Analysis via NLP & Semantic Competition Modeling

本專案以 **Google Play Store 上的語言學習 App** 為主要分析對象，透過語意文本探勘方法，探討 App 描述文本與其「存活機率」或「熱門程度」之間的關聯

此專案結合 **NLP、機器學習、實驗設計與研究導向分析流程** 

借鑒 Havakhor等人 (2023) 在創業融資領域的研究成果，該研究指出新創公司在社交媒體上的早期文本敘事具備顯著的訊號傳遞功能，能夠有效預測其募資成功率。


---

##  Research Questions (RQ)

### RQ1：描述文本的語意特徵是否能預測 App 是否存活？

* 使用 App description 與 review 萃取之語言與語意特徵
* 建立 Logistic Regression 模型
* 評估指標：Accuracy、ROC-AUC

---

### RQ2：與語意競爭者的差異程度是否影響 App 存活？

* 先以 K-Means 將 App 分成語意相近的 cluster（視為競爭群）
* 定義「語意差異」：

  * App 與其所屬 cluster 中心（特徵平均值）的距離
* 將距離作為新特徵，重新訓練存活預測模型

---

### RQ3：App 描述與使用者評論的語意一致性是否與存活有關？

* 計算 description 與 review embedding 的 cosine similarity
* 分析語意一致程度是否能預測存活

---

### RQ4：哪些語言特徵最能預測 App 存活？

* 同時納入多種語言特徵進行建模
* 透過 Logistic Regression 係數分析特徵重要性

---
## Survival Definition
標註準則：
* 存活=1：近 90 天內更新
* 存活=0：超過 90 天未更新


##  Features Used

* Text length (description / review)
* Unique word count
* Lexical diversity
* Average word length
* Description–Review cosine similarity
* Semantic distance to cluster center (RQ2)

所有特徵皆已正規化（standardized），以確保距離與模型穩定性。

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

## ▶️ How to Run

```bash
python main.py
```

執行後會：

* 依序完成 RQ1–RQ4
* 自動儲存模型評估結果（JSON）
* 自動儲存特徵重要性（CSV）

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

* 語意與語言特徵確實能在一定程度上預測 App 存活
* 與競爭群體「語意差異過大或過小」皆可能影響存活機率
* 描述與評論的語意一致性可視為產品定位清晰度的 proxy
* Logistic Regression 係數提供了良好的可解釋性

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
