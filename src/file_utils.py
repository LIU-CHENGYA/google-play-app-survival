import os
import json
import pandas as pd
### 載入資料
#function 1: 載入csv資料 回傳df(差在index_col="appId")
def load_df(file_name):
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR,'data',file_name)
    return pd.read_csv(DATA_DIR,index_col="appId")

#function 2: 載入csv資料 回傳df (差在index_col=0)
def load_df_2(file_name):
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR,'data',file_name)
    return pd.read_csv(DATA_DIR,index_col=0)

#function 3: 載入csv資料 回傳df (差在index_col="cluster")
# 專門load cluster分析結果檔案
def load_df_3(file_name):
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR,'data',file_name)
    return pd.read_csv(DATA_DIR,index_col="cluster")

### 儲存資料
#function 4: df存檔成csv (存到data資料夾裡面)
def save_file_csv(df,filename):
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    df.to_csv(os.path.join(DATA_DIR, filename),
            encoding="utf-8-sig",
            index=True)
#function 5: df存檔成csv (存到result_data資料夾裡面)
def save_file_csv_result(df,filename):
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "result_data")
    os.makedirs(DATA_DIR, exist_ok=True)

    df.to_csv(os.path.join(DATA_DIR, filename),
            encoding="utf-8-sig",
            index=True)

#function 6: 將dic 存成json (存到result_data資料夾裡面)
def save_file_json(date,filename):
    BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
    DATA_DIR = os.path.join(BASE_DIR,"result_data",filename)
    with open(DATA_DIR,'w',encoding="utf-8") as file:
        json.dump(date,file,ensure_ascii=True, indent=4)


