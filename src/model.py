#import
import pandas as pd
from src import file_utils

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


#function 1 訓練模型 :訓練一個「可以預測 App 是否存活」的模型
def train_survival_model (x_df,y_df):
    #切割資料
    x_train, x_test, y_train, y_test = train_test_split(x_df,y_df, test_size=0.2, random_state=1,stratify=y_df) #根據 y 的分布比例 來切分訓練集和測試集
    #建立Logistic模型
    model = LogisticRegression(max_iter=1000,random_state=0) #max_iter: 迭代次數，預設為100代 # random_state 固定隨機種子，讓模型結果可重現

    # 使用訓練資料訓練模型
    model.fit(x_train,y_train.values.ravel())
    return model,x_test,y_test


#function 2 模型是否有效
def evaluate_model(model,x_test,y_test,):
    # 用測試值做預測
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:,1] #存活的機率
    # 評估
    evaluate_result = {"Accuracy": model.score(x_test, y_test),
                    "ROC-AUC":roc_auc_score(y_test, y_prob),
                    "classification_report":classification_report(y_test, y_pred)}

    return evaluate_result

#function 3 哪些語言特徵有用
def get_feature_importance(model,feature_names):

    coef_df = pd.DataFrame({"feature":feature_names,"coef":model.coef_[0]}).sort_values("coef",ascending=False) #對應特徵的權重 #正值 → 該特徵越大，模型越傾向預測為「存活 (1)」 負值 → 該特徵越大，模型越傾向預測為「不存活 (0)」。

    return coef_df





if __name__ =='__main__':
    #載入兩個訓練資料
    train_x_df = file_utils.load_df_2("train_x.csv")
    train_y_df = file_utils.load_df_2("train_y.csv")
    #訓練模型
    model,x_test,y_test = train_survival_model(train_x_df,train_y_df)
    #評估模型
    evaluate_result = evaluate_model(model,x_test,y_test)
    #哪些語言特徵有用
    coef_df = get_feature_importance(model,train_x_df.columns)

