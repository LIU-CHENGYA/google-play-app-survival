# import 
from src import model
import numpy as np
from src import file_utils

def run_all():
    # 準備所需的data
    cosine_similarity_result = file_utils.load_df_2("cosine_similarity_result.csv")
    features_label = file_utils.load_df_2("features_labels.csv")
    x = file_utils.load_df_2("train_x.csv")
    y = file_utils.load_df_2("train_y.csv")
    cluster_analyze = file_utils.load_df_3("cluster_analyze.csv")

    # 回答 RQ1：描述文本語意特徵是否能預測 App 存活
    # 使用資料：train_x.csv, train_y.csv
    # 方法：Logistic Regression + ROC-AUC
    x_RQ1 = x
    y_RQ1 = y
    model_result,x_test,y_test = model.train_survival_model(x_RQ1,y_RQ1)
    evaluate_result_rq1 = model.evaluate_model(model_result,x_test,y_test)
    coef_df_rq1 = model.get_feature_importance(model_result,x.columns)
    #存檔: coef_df
    file_utils.save_file_json(evaluate_result_rq1,"evaluate_result_rq1.json")

    file_utils.save_file_csv_result(coef_df_rq1,"coef_df_rq1.csv")


    # 回答 RQ2 : 與競爭者的語意距離是否影響存活？
    # 在語意上「越不像同群競爭者」的 App，是否越容易存活
    distance_to_cluster_center = [] #App 在同語意競爭群中的「差異程度」
    for appId,row in features_label.iterrows():
        cluster_name = row['cluster'] #看它屬於哪個 cluster
        cosine_similarity_centroid = cluster_analyze.loc[cluster_name,"cosine_similarity"]
        desc_lexical_diversity_centroid = cluster_analyze.loc[cluster_name,"desc_lexical_diversity"]
        rev_text_length_centroid = cluster_analyze.loc[cluster_name,"rev_text_length"]
        distance_to_cluster_center.append (np.sqrt((row["cosine_similarity"]-cosine_similarity_centroid)**2 +
                                                    (row["desc_lexical_diversity"]-desc_lexical_diversity_centroid)**2 +
                                                    (row["rev_text_length"]- rev_text_length_centroid)**2 ))
    features_label['distance_to_cluster_center'] = distance_to_cluster_center
    #模型

    x_RQ2 = features_label[['distance_to_cluster_center']]
    y_RQ2 = y

    model_result,x_test,y_test = model.train_survival_model(x_RQ2,y_RQ2)
    evaluate_result_rq2 = model.evaluate_model(model_result,x_test,y_test)
    coef_df_rq2 = model.get_feature_importance(model_result,['distance_to_cluster_center'])
    file_utils.save_file_json(evaluate_result_rq2,"evaluate_result_rq2.json")
    file_utils.save_file_csv_result(coef_df_rq2,"coef_df_rq2.csv")


    #回答 RQ3 : 描述文本與使用者評論的語意相似度是否與存活有關？
    x_RQ3 = cosine_similarity_result
    y_RQ3 = y
    model_result,x_test,y_test = model.train_survival_model(x_RQ3,y_RQ3)
    evaluate_result_rq3 = model.evaluate_model(model_result,x_test,y_test)
    coef_df_rq3 = model.get_feature_importance(model_result,['cosine_similarity'])
    file_utils.save_file_json(evaluate_result_rq3,"evaluate_result_rq3.json")
    file_utils.save_file_csv_result(coef_df_rq3,"coef_df_rq3.csv")

    #回答 RQ4 哪些語言特徵最能預測存活或熱門程度？
    x_RQ4 = features_label[["desc_text_length",	"desc_unique_word_count","desc_lexical_diversity","desc_avg_word_length",	"rev_text_length","rev_unique_word_count",	"rev_lexical_diversity"	,"rev_avg_word_length","cosine_similarity"]]
    y_RQ4 = y
    model_result, x_test, y_test = model.train_survival_model(x_RQ4, y_RQ4)
    evaluate_result_rq4 = model.evaluate_model(model_result, x_test, y_test)
    coef_df_rq4 = model.get_feature_importance(model_result, x_RQ4.columns)
    file_utils.save_file_json( evaluate_result_rq4,"evaluate_result_rq4.json")
    file_utils.save_file_csv_result(coef_df_rq4, "coef_df_rq4.csv")
