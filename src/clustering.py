#import
import file_utils
import pandas as pd

import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler



# function 1 將特徵合併
def feature_merge(df1,df2,df3):
    ##先對齊appId
    common_appId = df1.index.intersection(df2.index)

    #取出共同的index
    df1_common_appId = df1.loc[common_appId]  
    df2_common_appId = df2.loc[common_appId]

    #把需要的欄位分別取出來
    df1_filter_features = df1_common_appId[['text_length','unique_word_count','lexical_diversity','avg_word_length']]
    df2_filter_features = df2_common_appId[['text_length','unique_word_count','lexical_diversity','avg_word_length']]

    #更改兩個表格欄位名稱
    df1_filter_features.columns = ['desc_text_length','desc_unique_word_count','desc_lexical_diversity','desc_avg_word_length']
    df2_filter_features.columns = ['rev_text_length','rev_unique_word_count','rev_lexical_diversity','rev_avg_word_length']
    #正式合併
    result = pd.concat([df1_filter_features, df2_filter_features], axis=1) #合併

    #再合併cosine_similary

    #再對齊一次appId
    common_appId = result.index.intersection(df3.index)
    #取出共同的index
    result_filter = result.loc[common_appId]  
    df3_filter = df3.loc[common_appId]
    final = pd.concat([result_filter,df3_filter],axis=1)

    #整理表格資料並標準化
    final =final.fillna(0)
    return final


# function 2 對features_result 做標準化
# 使用Min-Max Scaling (最小最大值縮放)
def standard_df(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)
    

# function 3 算不同 k 的評估指標
## 回傳 inertia / silhouette list
def evaluate_kmeans_k(df):
    x = df[["cosine_similarity","desc_lexical_diversity","rev_text_length"]]

    silhouette_list = []
    inertia_list = []

    for k in range(2,13):
        model = KMeans(n_clusters=k,n_init='auto',random_state=1) #random_state 是為了固定隨機種子，使結果可以重現
        model.fit(x)
        score1 = silhouette_score(x, model.labels_) # 側影函數驗證數據集群內一致性的方法
        score2 = model.inertia_  #求出每個Cluster內的資料與其中心點之平方距離和 #越大代表越差
        silhouette_list.append([k,score1])
        inertia_list.append([k,score2])
    
    silhouette_df = pd.DataFrame(silhouette_list,columns=["k","score"])
    inertia_df = pd.DataFrame(inertia_list,columns=["k","score"])
    return silhouette_df,inertia_df


#function 4 建立模型
#準備做k-means 選擇的特徵欄位為: cosine_similarity、desc_lexical_diversity、rev_text_length
def k_means_model(df,n):
    x = df[["cosine_similarity","desc_lexical_diversity","rev_text_length"]]
    model = KMeans(n_clusters=n,n_init='auto',random_state=1) #random_state 是為了固定隨機種子，使結果可以重現
    model.fit(x)
    df['cluster'] = model.labels_
    return df

#function 5 做分群處理:summarize each cluster
#每一群的「平均特徵」
def cluster_analyze(df_add_culster):
    analyze_cluster = df_add_culster.groupby('cluster').mean()
    return analyze_cluster



    
if __name__ == "__main__":
    #1. load資料
    metadata_features = file_utils.load_df_2("metadata_add_feature.csv")
    revies_features = file_utils.load_df_2("review_add_feature.csv")
    cos_sim = file_utils.load_df_2("cosine_similarity_result.csv")

    #2. 將特徵合併
    features_df = feature_merge(metadata_features,revies_features,cos_sim)

    #3. 對features_result 做標準化
    standard_features = standard_df(features_df)
    #存一次檔
    file_utils.save_file_csv(standard_features,'standard_features.csv')

    #4. 評估k值
    silhouette_result,inertia_result = evaluate_kmeans_k(standard_features)
    #存檔
    file_utils.save_file_csv(silhouette_result,'silhouette_df.csv')
    file_utils.save_file_csv(inertia_result,'inertia_df.csv')

    #5. 建模型(加上了label)
    featuers_add_labels = k_means_model(standard_features,3)
    #存檔(加上了label)
    file_utils.save_file_csv(featuers_add_labels,'features_labels.csv')

    #6 分群分析
    analyze_cluster = cluster_analyze(featuers_add_labels)
    #存檔
    file_utils.save_file_csv(analyze_cluster,'cluster_analyze.csv')
