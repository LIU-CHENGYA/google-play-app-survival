#import
import file_utils

#function 1: 目的是將metadata與存活資訊對齊後輸出 輸出x,y
def survial_df(meta_df,sta_feature_df):
    common_appId = meta_df.index.intersection(sta_feature_df.index)
    df1 = meta_df.loc[common_appId]
    df2 = sta_feature_df.loc[common_appId]
    train_x = df2
    train_y = df1["survive"]
    return train_x,train_y


if __name__ =='__main__':
    meta = file_utils.load_df_2("app_metadata_clean.csv")
    sta_feature_df = file_utils.load_df_2("standard_features.csv")
    train_x,train_y = survial_df(meta,sta_feature_df)

    file_utils.save_file_csv(train_x,"train_x.csv")
    file_utils.save_file_csv(train_y,"train_y.csv")

