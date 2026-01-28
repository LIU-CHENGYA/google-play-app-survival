# import
import pandas as pd
import numpy as np
import file_utils
from collections import Counter


# function 1 建立字典
def build_voc(df_clean,column):
    voc = []
    for content in df_clean[column]:
        for term in content.split():
            voc.append(term)
    voc_final = set(voc)
    return voc_final

# function 2 建立tf_idf
def build_tf_idf(df_clean,column): #建立tf-idf 時就應該有index = appId
    all_app_counts = {}

    for appId,row in df_clean.iterrows(): #每一 row
        content = row[column] #文字
        counts = Counter(content.split()) #每個 App 算好它的詞頻小字典
        total_lengh = len(content.split())
        for k,v in counts.items():
            counts[k] = v/total_lengh
        
        all_app_counts[appId] = counts

    tf = pd.DataFrame(all_app_counts) #用 Pandas 一口氣對齊所有單字
    tf = tf.fillna(0)


    num_app =  tf.shape[1] #計算有幾欄(有幾個app)
    df = (tf > 0).sum() #加總
    idf = np.log(num_app+1/(df+1))

    tf_idf = tf * idf
    df_idf = tf_idf.T
    return df_idf

# function 3 cosine_similarity (每一個 app，自行算自己的 similarity)
## 一次回傳「所有 app 的 cosine similarity」
### 得到的 cosine similarity Series，每一個值代表：「某個 app 的 description 與它自己的 review 的語意相似度」。
#### 「同一個 app 的兩種文本（描述 vs 評論）」之間的相似度。

def bulid_cosine_similarity(tf_idf_1,tf_idf_2):
    # 先對齊appId
    common_app = tf_idf_1.index.intersection(tf_idf_2.index) # 因為研究問題是「描述文本 vs 使用者評論」的語意相似度，代表這個app必須同時有 description 與 review
    X = tf_idf_1.loc[common_app]
    Y = tf_idf_2.loc[common_app]

    # 再對齊「詞彙空間」
    X_aligned, Y_aligned = X.align(Y, join="outer", axis=1, fill_value=0)

    # 內積(存成serise)
    inner_product = (X_aligned * Y_aligned).sum(axis=1)
    
    
    #各自長度
    norm_x = np.sqrt((X_aligned **2 ).sum(axis=1))
    norm_y = np.sqrt((Y_aligned **2 ).sum(axis=1))

    cos_sim_result = inner_product/(norm_x* norm_y)

    #加上欄位名稱
    cos_sim_result.name = 'cosine_similarity'
    return cos_sim_result 

# function 4 增加文本的語言特徵
### .str 是一個向量化字串操作的存取器 (accessor)，
### 它允許你對 DataFrame 或 Series 中的每一列（如果該列是字串类型）應用各種字串方法
def add_feature(df,column):
    # 文字長度
    df['text_length'] = df[column].str.split().str.len()

    # 詞彙多樣性
    df['unique_word_count'] = df[column].str.split().apply(lambda x: len(set(x)))

    # 詞彙多樣性比例
    df['lexical_diversity'] = df['unique_word_count'] /df['text_length']

    # 平均詞長 每個單字平均有幾個字母 (看專業度)
    ## apply(...) 是「對每一列做一次」
    df['avg_word_length'] = df[column].str.split().apply(
                                lambda x:sum(len(w) for w in x) / len(x) if len(x) > 0 else 0)
    return df





if __name__ =="__main__":
    # 1. load資料
    df_metadata_clean = file_utils.load_df('app_metadata_clean.csv')
    df_reviewmerge_clean = file_utils.load_df('app_review_merge_clean.csv')

    # 2. 建立tf_idf
    tf_idf_metadata = build_tf_idf(df_metadata_clean,"clean_description")
    tf_idf_review = build_tf_idf(df_reviewmerge_clean,"clean_review")
    # 3. 計算cosine_similarity
    cos_sim_result = bulid_cosine_similarity(tf_idf_metadata,tf_idf_review)
    print(f'計算cosine_similarity: {cos_sim_result}')
    # 4. 增加特徵
    df_metadata_add_feature = add_feature(df_metadata_clean,"clean_description")
    df_review_add_feature = add_feature(df_reviewmerge_clean,"clean_review")

    # 5. 存檔
    file_utils.save_file_csv(df_metadata_add_feature,'metadata_add_feature.csv')
    file_utils.save_file_csv(df_review_add_feature,'review_add_feature.csv')
    file_utils.save_file_csv(cos_sim_result,'cosine_similarity_result.csv')