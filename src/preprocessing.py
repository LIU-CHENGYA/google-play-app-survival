#import
import re
import file_utils
import pandas as pd
# 建立 PorterStemmer 物件
from nltk.stem.porter import PorterStemmer


# function 1 :前處理 (共用)
def preprocess_text(df,col): #col是字串(欄位名稱)
    ps = PorterStemmer()
    new_column = []
    # 在進入迴圈之前，先確保 df["review"] 裡面全部都是字串
    df = df.dropna(subset=[col]).reset_index() #直接刪除 review 是空的那些資料列，把號碼牌重新發一次

    stop_words = {'i','it','the','a','an','is','am','in','at','of','on','and','or','to','for','app','you','your','with'}

    #迴圈處理
    for content in df[col]:
        text = re.findall(r'[a-z]+',content.lower()) #結果會是list
        stem_token = [ps.stem(t) for t in text]
        
        filter_token = [t for t in stem_token if t not in stop_words and t != '']
        new_column.append(" ".join(filter_token)) #只存字串不存 []concat 

    # 新增欄位到原本的df上面
    df['clean_'+ col] = new_column

    return df

# function 2 合併(針對review)
def merge_review(df):
    df_merge_review = df.groupby('appId')['clean_review'].apply(lambda x : ' '. join(x)).reset_index()
    return df_merge_review


if __name__ == "__main__":
    #1. load檔案
    df1 = file_utils.load_df('app_metadata.csv')
    df2 = file_utils.load_df('app_review.csv')

    #2. 前處理
    df_metadata_clean = preprocess_text(df1,'description')
    df_review_clean = preprocess_text(df2,'review')

    #3. review 合併
    df_merge_review = merge_review(df_review_clean)

    #4. 存檔
    file_utils.save_file_csv(df_metadata_clean,'app_metadata_clean.csv')
    file_utils.save_file_csv(df_merge_review,'app_review_merge_clean.csv')