import os
import pandas as pd
import numpy as np
import random
import gc
from sklearn import preprocessing
np.random.seed(2019)
random.seed(2019)

def norm(train_df,test_df,features):   
    df=pd.concat([train_df,test_df])[features]
    scaler = preprocessing.QuantileTransformer(random_state=0)
    scaler.fit(df[features]) 
    train_df[features]=scaler.transform(train_df[features])
    test_df[features]=scaler.transform(test_df[features])


for path1,path2,flag,wday in [('data/train_dev.pkl','data/dev.pkl','dev',1),('data/train.pkl','data/test.pkl','test',3)]:
        print(path1,path2)
        train_df=pd.read_pickle(path1)
        test_df=pd.read_pickle(path2)
        train_df=train_df.sort_values(by=['aid','request_day'])
        train_df=train_df.drop_duplicates(keep='last')
        print(train_df.shape,test_df.shape)
        test_df['wday']=wday
        float_features=['history_aid_imp', 'history_aid_bid', 'history_aid_pctr', 'history_aid_quality_ecpm', 
                        'history_aid_totalEcpm', 
                        'periods_cont','aid_imp_mean', 'aid_imp_median', 'aid_imp_std', 'aid_imp_min',
                        'aid_imp_max', 'good_id_imp_mean', 'good_id_imp_median',
                        'good_id_imp_std', 'good_id_imp_min', 'good_id_imp_max', 'advertiser_imp_mean', 
                        'advertiser_imp_median', 'advertiser_imp_std', 'advertiser_imp_min', 'advertiser_imp_max', 
                        'good_id_advertiser_count', 'good_id_aid_count', 'good_id_ad_size_count', 
                        'good_id_ad_type_id_count', 'good_id_good_id_size', 'advertiser_good_id_count', 
                        'advertiser_aid_count', 'advertiser_ad_size_count', 'advertiser_ad_type_id_count',
                        'advertiser_good_type_count', 'advertiser_advertiser_size', 
                        'good_type_good_type_size', 'aid_aid_size','create_timestamp'] 
        train_df[float_features]=train_df[float_features].fillna(0)
        test_df[float_features]=test_df[float_features].fillna(0)
        norm(train_df,test_df,float_features)
        
        print(train_df[float_features])
        
        k=1
        train_df=train_df.sample(frac=1)
        train=[(path2[:-4]+'_NN.pkl',test_df)]
        for i in range(k):
            train.append((path1[:-4]+'_NN_'+str(i)+'.pkl',train_df.iloc[int(i/k*len(train_df)):int((i+1)/k*len(train_df))]))
        del train_df
        gc.collect()
        for file,temp in train:
            print(file,temp.shape)
            print("w2v")
            for pivot,f,L in [('uid','aid',64),('uid','good_id',64),('uid','advertiser',64)]:
                df = pd.read_pickle('data/' +pivot+'_'+ f +'_'+flag +'_w2v_'+str(L)+'.pkl')                                 
                if 'train' in file:
                    items=[]
                    for item in temp[f].values:
                        if random.random()<0.05:
                            items.append(-11111111111)
                        else:
                            items.append(item)
                    temp['tmp']=items
                    df['tmp']=df[f]
                    del df[f]
                    temp = pd.merge(temp, df, on='tmp', how='left')
                else:
                    temp = pd.merge(temp, df, on=f, how='left')
                print(temp.shape)
                     
            print('deepwalk')
            for f1,f2,L in [('uid','aid',64),('uid','good_id',64)]:
                df = pd.read_pickle('data/' +f1+'_'+ f2+'_'+f2 +'_'+flag +'_deepwalk_'+str(L)+'.pkl')  
                if 'train' in file:
                    items=[]
                    for item in temp[f2].values:
                        if random.random()<0.05:
                            items.append(-11111111111)
                        else:
                            items.append(item)
                    temp['tmp']=items
                    df['tmp']=df[f2]
                    del df[f2]
                    temp = pd.merge(temp, df, on='tmp', how='left')
                else:
                    temp = pd.merge(temp, df, on=f2, how='left')
                print(temp.shape) 
                
            temp=temp.fillna(0)           
            temp.to_pickle(file)
            del temp
            gc.collect()