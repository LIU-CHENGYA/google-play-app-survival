#import
import pandas as pd

from google_play_scraper import app
from google_play_scraper import search
from google_play_scraper.exceptions import NotFoundError
from google_play_scraper import Sort, reviews


import datetime
from datetime import date
from datetime import datetime
import file_utils


# function 1：抓 App ID
def retrieve_appId():
    key_category = ['language learning','Education','vocabulary','English','listen','Grammar','TOEIC','IELTS','janpanese','teaching','speak','French','AI learning']
    key_appId = []
    
    for category in key_category:
        result = search(
        category,
        lang="en",  # defaults to 'en'
        country="us",  # defaults to 'us'
        n_hits=30  # defaults to 30 (= Google's maximum)
        )
        for x in result:
            key_appId.append(x['appId'])
    key_appId_only = set(key_appId)  
    return  key_appId_only


# function 2：抓 App 詳細資訊
def app_metadata(key_appId_only):

    data = []

    now = datetime.now()
    date_now = now.date()

    for id in key_appId_only:
        try:
            result = app(
            id,
            lang='en', # defaults to 'en'
            country='us' # defaults to 'us'
            )
            #處理時間搓印 -> days_since_update
            if result['updated'] == '' or result['updated'] == None:
                days_since_update = None
            else: 
                seconds_timestamp = int(result['updated'])
                dt_object = datetime.fromtimestamp(seconds_timestamp)
                date_last_updated = dt_object.date()


                delta = date_now - date_last_updated
                days_since_update = delta.days

            #處理survive 這個欄位:days_since_update ≤ 90 → 存活 = 1
            SURVIVAL_THRESHOLD_DAYS = 90
            if  days_since_update is None or days_since_update > SURVIVAL_THRESHOLD_DAYS :
                survive = 0
            else:
                survive = 1     

            #rating 有多少人按過星星 用來表示使用者參與情況
            data.append([result['appId'],result['title'],result['description'],result['realInstalls'],result['score'],result['ratings'],result['categories'],days_since_update,survive])
        except NotFoundError:
            print(f"App not found: {id}")
            continue
    
    df = pd.DataFrame(data)
    df.columns = ["appId",
    "title",
    "description",
    "realInstalls",
    "score",
    "ratings",
    "categories",
    "days_since_update",
    "survive"
    ]    

    return df


# function 3：抓評論
def app_review(key_appId_only):
    data_review = []
    
    for id in key_appId_only :
        try:
            result, continuation_token = reviews(
                id,
                lang='en', # defaults to 'en'
                country='us', # defaults to 'us'
                sort=Sort.NEWEST, # defaults to Sort.NEWEST / 最新發表的評論
                count=30, # defaults to 100
            )
            for x in result:
                data_review.append([id,x['content']])
        except NotFoundError:
            print(f"review not found: {id}")
            continue
    
    df = pd.DataFrame(data_review)
    df.columns = ["appId",
        "review",
        ]
        
    return df


if __name__ == "__main__":
    # 1. 取得 App ID
    key_appId_only = retrieve_appId()
    
    # 2. 抓 App 詳細資訊
    df_metadata = app_metadata(key_appId_only)
    # 將詳細資訊存檔
    file_utils.save_file_csv(df_metadata,"app_metadata.csv")
    
    # 3. 抓評論
    df_review = app_review(key_appId_only)
    # 將評論存檔
    file_utils.save_file_csv(df_review,"app_review.csv")
