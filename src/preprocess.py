import os
import pandas as pd
import numpy as np
import random
import gc
import time
from tqdm import tqdm

def parse_rawdata():
    #曝光日志
    df=pd.read_csv('data/testA/totalExposureLog.out', sep='\t',names=['id','request_timestamp','position','uid','aid','imp_ad_size','bid','pctr','quality_ecpm','totalEcpm']).sort_values(by='request_timestamp')
    df[['id','request_timestamp','position','uid','aid','imp_ad_size']]=df[['id','request_timestamp','position','uid','aid','imp_ad_size']].astype(int)
    df[['bid','pctr','quality_ecpm','totalEcpm']]=df[['bid','pctr','quality_ecpm','totalEcpm']].astype(float) 
    df.to_pickle('data/testA/totalExposureLog.pkl') 
    del df
    gc.collect()
    ##############################################################################
    #静态广告
    df =pd.read_csv('data/testA/ad_static_feature.out', sep='\t', names=['aid','create_timestamp','advertiser','good_id','good_type','ad_type_id','ad_size']).sort_values(by='create_timestamp')
    df=df.fillna(-1)
    for f in ['aid','create_timestamp','advertiser','good_id','good_type','ad_type_id']:
        items=[]
        for item in df[f].values:
            try:
                items.append(int(item))
            except:
                items.append(-1)
        df[f]=items
        df[f]=df[f].astype(int)
    df['ad_size']=df['ad_size'].apply(lambda x:' '.join([str(int(float(y))) for y in str(x).split(',')]))    
    df.to_pickle('data/testA/ad_static_feature.pkl')
    del df
    gc.collect()
    ##############################################################################
    #用户信息
    df =pd.read_csv('data/testA/user_data', sep='\t', 
                  names=['uid','age','gender','area','status','education','concuptionAbility','os','work','connectionType','behavior'])
    df=df.fillna(-1)
    df[['uid','age','gender','education','consuptionAbility','os','connectionType']]=df[['uid','age','gender','education','concuptionAbility','os','connectionType']].astype(int)
    for f in ['area','status','work','behavior']:
        df[f]=df[f].apply(lambda x:' '.join(x.split(',')))
    df.to_pickle('data/testA/user_data.pkl')
    del df
    gc.collect()
    ##############################################################################
    #测试数据
    df=pd.read_csv('data/testA/test_sample.dat', sep='\t', names=['id','aid','create_timestamp','ad_size','ad_type_id','good_type','good_id','advertiser','delivery_periods','crowd_direction','bid'])
    df=df.fillna(-1)
    df[['id','aid','create_timestamp','ad_size','ad_type_id','good_type','good_id','advertiser']]=df[['id','aid','create_timestamp','ad_size','ad_type_id','good_type','good_id','advertiser']].astype(int)
    df['bid']=df['bid'].astype(float)
    df.to_pickle('data/testA/test_sample.pkl')
    del df
    gc.collect()
    ##############################################################################
    #广告操作数据
    df=pd.read_csv('data/testA/ad_operation.dat', sep='\t',names=['aid','request_timestamp','type','op_type','value']).sort_values(by='request_timestamp')
    df['request_time'] = df.apply(lambda x:(pd.to_datetime('20190228000000') if x['request_timestamp'] == 20190230000000  else (pd.to_datetime(x['request_timestamp']) if x['request_timestamp'] == 0 else pd.to_datetime(str(x['request_timestamp'])))), axis=1 )
    df.to_pickle('data/testA/ad_operation.pkl')

def construct_log():
    #构造曝光日志，分别有验证集的log和测试集的log
    train_df=pd.read_pickle('data/testA/totalExposureLog.pkl')
    train_df['request_day']=train_df['request_timestamp']//(3600*24)
    wday=[]
    hour=[]
    minute=[]
    for x in tqdm(train_df['request_timestamp'].values,total=len(train_df)):
        localtime=time.localtime(x)
        wday.append(localtime[6])
        hour.append(localtime[3])
        minute.append(localtime[4])
    train_df['wday']=wday
    train_df['hour']=hour
    train_df['minute']=minute
    train_df['period_id']=train_df['hour']*2+train_df['minute']//30
    dev_df=train_df[train_df['request_day']==17974]
    del dev_df['period_id']
    del dev_df['minute']
    del dev_df['hour']
    log=train_df
    tmp = pd.DataFrame(train_df.groupby(['aid','request_day']).size()).reset_index()
    tmp.columns=['aid','request_day','imp']
    log=log.merge(tmp,on=['aid','request_day'],how='left')
    log[log['request_day']<17973].to_pickle('data/user_log_dev.pkl')
    log.to_pickle('data/user_log_test.pkl')
    del log
    del tmp
    gc.collect()
    del train_df['period_id']
    del train_df['minute']
    del train_df['hour']    
    return train_df,dev_df
    
def extract_setting():
    aids=[]
    with open('data/testA/ad_operation.dat','r') as f:
        for line in f:
            line=line.strip().split('\t')
            try:
                if line[1]=='20190230000000':
                    line[1]='20190301000000'
                if line[1]!='0':
                    request_day=time.mktime(time.strptime(line[1], '%Y%m%d%H%M%S'))//(3600*24)
                else:
                    request_day=0
            except:
                print(line[1])

            if len(aids)==0:
                aids.append([int(line[0]),0,"NaN","NaN"])
            elif aids[-1][0]!=int(line[0]):
                for i in range(max(17930,aids[-1][1]+1),17975):
                    aids.append(aids[-1].copy())
                    aids[-1][1]=i
                aids.append([int(line[0]),0,"NaN","NaN"])               
            elif request_day!=aids[-1][1]:
                for i in range(max(17930,aids[-1][1]+1),int(request_day)):
                    aids.append(aids[-1].copy())
                    aids[-1][1]=i                
                aids.append(aids[-1].copy())
                aids[-1][1]=int(request_day)
            if line[3]=='3':
                aids[-1][2]=line[4]
            if line[3]=='4':
                aids[-1][3]=line[4]
    ad_df=pd.DataFrame(aids)
    ad_df.columns=['aid','request_day','crowd_direction','delivery_periods']
    return ad_df
    
def construct_train_data(train_df):
    #构造训练集
    #算出广告当天平均出价和曝光量
    tmp = pd.DataFrame(train_df.groupby(['aid','request_day'])['bid'].nunique()).reset_index()
    tmp.columns=['aid','request_day','bid_unique']
    train_df=train_df.merge(tmp,on=['aid','request_day'],how='left')
    tmp = pd.DataFrame(train_df.groupby(['aid','request_day']).size()).reset_index()
    tmp_1 = pd.DataFrame(train_df.groupby(['aid','request_day'])['bid'].mean()).reset_index()
    tmp.columns=['aid','request_day','imp']
    del train_df['bid']
    tmp_1.columns=['aid','request_day','bid']
    train_df=train_df.drop_duplicates(['aid','request_day'])
    train_df=train_df.merge(tmp,on=['aid','request_day'],how='left')
    train_df=train_df.merge(tmp_1,on=['aid','request_day'],how='left')
    del tmp
    del tmp_1
    gc.collect()
    #去重，得到训练集
    train_df=train_df.drop_duplicates(['aid','request_day'])
    del train_df['request_timestamp']
    del train_df['uid']

    #过滤未出现在广告操作文件的广告
    ad_df=extract_setting()
    ad_df=ad_df.drop_duplicates(['aid','request_day'],keep='last')
    ad_df['request_day']+=1
    train_df=train_df.merge(ad_df,on=['aid','request_day'],how='left')
    train_df['is']=train_df['crowd_direction'].apply(lambda x:type(x)==str)
    train_df=train_df[train_df['is']==True]
    train_df=train_df[train_df['crowd_direction']!="NaN"]
    train_df=train_df[train_df['delivery_periods']!="NaN"]

    #过滤出价和曝光过高的广告
    train_df=train_df[train_df['imp']<=3000]
    train_df=train_df[train_df['bid']<=1000]
    train_dev_df=train_df[train_df['request_day']<17973]
    print(train_df.shape,train_dev_df.shape)
    print(train_df['imp'].mean(),train_df['bid'].mean())
    return train_df,train_dev_df

def construct_dev_data(dev_df):
    #构造验证集
    #过滤掉当天操作的广告，和未出现在操作日志的广告
    aids=set()
    exit_aids=set()
    with open('data/testA/ad_operation.dat','r') as f:
        for line in f:
            line=line.strip().split('\t')
            if line[1]=='20190230000000':
                line[1]='20190301000000'
            if line[1]!='0':
                request_day=time.mktime(time.strptime(line[1], '%Y%m%d%H%M%S'))//(3600*24)
            else:
                request_day=0
            if request_day==17974:
                aids.add(int(line[0]))
            exit_aids.add(int(line[0]))
    dev_df['is']=dev_df['aid'].apply(lambda x: x in aids)
    dev_df=dev_df[dev_df['is']==False]
    dev_df['is']=dev_df['aid'].apply(lambda x: x in exit_aids)
    dev_df=dev_df[dev_df['is']==True]
    #过滤当天出价不唯一的广告
    tmp = pd.DataFrame(dev_df.groupby('aid')['bid'].nunique()).reset_index()
    tmp.columns=['aid','bid_unique']
    dev_df=dev_df.merge(tmp,on='aid',how='left')
    dev_df=dev_df[dev_df['bid_unique']==1]
    #统计广告当天的曝光量
    tmp = pd.DataFrame(dev_df.groupby('aid').size()).reset_index()
    tmp.columns=['aid','imp']
    dev_df=dev_df.merge(tmp,on='aid',how='left')
    dev_df=dev_df.drop_duplicates('aid')
    #过滤未出现在广告操作文件的广告
    ad_df=extract_setting()
    ad_df=ad_df.drop_duplicates(['aid'],keep='last')
    dev_df=dev_df.merge(ad_df,on='aid',how='left')
    dev_df=dev_df[dev_df['crowd_direction']!="NaN"]
    dev_df=dev_df[dev_df['delivery_periods']!="NaN"].reset_index()
    del dev_df['index']
    del dev_df['request_timestamp']
    del dev_df['is']
    del dev_df['uid']
    #构建虚假广告，测试单调性
    items=[]
    for item in dev_df[['aid','bid','crowd_direction', 'delivery_periods','imp']].values:
        item=list(item)
        items.append(item+[1])
        for i in range(10):
            while True:
                t=random.randint(0,2*item[1])
                if t!=item[1]:
                    items.append(item[:1]+[t]+item[2:]+[0])
                    break
                else:
                    continue
    dev_df=pd.DataFrame(items)
    dev_df.columns=['aid', 'bid', 'crowd_direction', 'delivery_periods','imp','gold'] 
    del items
    gc.collect()
    print(dev_df.shape)
    print(dev_df['imp'].mean(),dev_df['bid'].mean())
    return dev_df


print("parsing raw data ....")
parse_rawdata()

print("construct log ....")
train_df,dev_df=construct_log()

print("construct train data ....")
train_df,train_dev_df=construct_train_data(train_df)

print("construct dev data ....")
dev_df=construct_dev_data(dev_df)

print("load test data ....")
test_df=pd.read_pickle('data/testA/test_sample.pkl')

print("combine advertise features ....")
ad_df =pd.read_pickle('data/testA/ad_static_feature.pkl')
train_df=train_df.merge(ad_df,on='aid',how='left')
train_dev_df=train_dev_df.merge(ad_df,on='aid',how='left')
dev_df=dev_df.merge(ad_df,on='aid',how='left')

print("save preprocess data ....")
train_dev_df.to_pickle('data/train_dev.pkl')
train_df.to_pickle('data/train.pkl')
dev_df.to_pickle('data/dev.pkl')
test_df.to_pickle('data/test.pkl')
print(train_dev_df.shape,dev_df.shape)
print(train_df.shape,test_df.shape)
