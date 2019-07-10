import numpy as np
import pandas as pd
import ctrNet
import tensorflow as tf
from src import misc_utils as utils
import os
import gc
from sklearn import metrics
from sklearn import preprocessing
import random
np.random.seed(2019)


####################################################################################

#单值特征，直接embedding:支持任何可以转换成字符串的数据类型，比如浮点数，会转换成字符串
single_features=['periods_cont','aid','gender','crowd_direction', 'delivery_periods','advertiser', 'good_id', 'good_type', 'ad_type_id','consuptionAbility', 'os', 'work','connectionType','ad_size', 'good_id_advertiser_count', 'good_id_aid_count', 'good_id_ad_size_count', 'good_id_ad_type_id_count', 'good_id_good_id_size', 'advertiser_good_id_count', 'advertiser_aid_count', 'advertiser_ad_size_count', 'advertiser_ad_type_id_count', 'advertiser_good_type_count', 'advertiser_advertiser_size',]

#交叉特征，会使用分解机提取特征:支持任何可以转换成字符串的数据类型。比如浮点数，会转换成字符串
cross_features=[ 'aid','gender','crowd_direction', 'delivery_periods','advertiser', 'good_id', 'good_type', 'ad_type_id','consuptionAbility', 'os','work','connectionType','ad_size']

#多值特征，会使用分解机提取特征:支持字符串数据类型，用空格隔开
multi_features=['aid_uids','age','area','status','behavior','good_id_advertisers','good_id_request_days', 'good_id_positions', 'good_id_period_ids','good_id_wdays','advertiser_good_ids', 'advertiser_request_days', 'advertiser_positions',  'advertiser_period_ids', 'advertiser_wdays']

#稠密特征，直接放入MLP中:主要用于embedding特征，转化率等
dense_features=['uid_w2v_embedding_aid_64_'+str(i) for i in range(64)]+['uid_w2v_embedding_good_id_64_'+str(i) for i in range(64)]+['uid_w2v_embedding_advertiser_64_'+str(i) for i in range(64)]
dense_features+=['uid_aid_aid_deepwalk_embedding_64_'+str(i) for i in range(64)]+['uid_good_id_good_id_deepwalk_embedding_64_'+str(i) for i in range(64)]
dense_features+=['periods_on_'+str(i) for i in range(48)]

#key-values 特征，将稠密特征转换成向量: 浮点数类型，且数值在[0,1]之间
kv_features=['history_aid_imp', 'history_aid_bid', 'history_aid_pctr', 'history_aid_quality_ecpm', 'history_aid_totalEcpm',  'good_id_advertiser_count', 'good_id_aid_count', 'good_id_ad_size_count',  'good_id_ad_type_id_count', 'good_id_good_id_size', 'advertiser_good_id_count','advertiser_aid_count', 'advertiser_ad_size_count', 'advertiser_ad_type_id_count', 'advertiser_good_type_count', 'advertiser_advertiser_size', 'good_id_imp_median', 'good_id_imp_std', 'good_id_imp_min', 'good_id_imp_max', 'advertiser_imp_mean', 'advertiser_imp_median',  'advertiser_imp_std', 'advertiser_imp_min', 'advertiser_imp_max','create_timestamp']

####################################################################################

#参数
hparam=tf.contrib.training.HParams(
            model='CIN',
            norm=True,
            batch_norm_decay=0.9,
            hidden_size=[1024,512],
            dense_hidden_size=[300],
            cross_layer_sizes=[128,128],
            k=16,
            single_k=16,
            max_length=100,
            cross_hash_num=int(5e6),
            single_hash_num=int(5e6),
            multi_hash_num=int(1e6),
            batch_size=32,
            infer_batch_size=2**14,
            optimizer="adam",
            dropout=0,
            kv_batch_num=10,
            learning_rate=0.0002,
            num_display_steps=1000,
            num_eval_steps=1000,
            epoch=1, #don't modify
            metric='SMAPE',
            activation=['relu','relu','relu'],
            init_method='tnormal',
            cross_activation='relu',
            init_value=0.001,
            single_features=single_features,
            cross_features=cross_features,
            multi_features=multi_features,
            dense_features=dense_features,
            kv_features=kv_features,
            label='imp',
            model_name="CIN")
utils.print_hparams(hparam)

####################################################################################

#读取数据
test=pd.read_pickle('data/test_NN.pkl')
dev=pd.read_pickle('data/dev_NN.pkl')
train=pd.read_pickle('data/train_NN_0.pkl')
train_dev=pd.read_pickle('data/train_dev_NN_0.pkl')
train['gold_imp']=train['imp']
dev['gold_imp']=dev['imp']
train['imp']=train['imp'].apply(lambda x:np.log(x+1))
train_dev['imp']=train_dev['imp'].apply(lambda x:np.log(x+1))

####################################################################################

#检验模型
print(dev.shape)
print(train_dev.shape)
scaler=preprocessing.MinMaxScaler(feature_range=(0,8))
scaler.fit(train_dev[['imp']]) 
hparam.train_scaler=scaler
hparam.test_scaler=scaler
print("*"*80)
model=ctrNet.build_model(hparam)
model.train(train_dev,dev)
dev_preds=np.zeros(len(dev))
dev_preds=model.infer(dev)
dev_preds=np.exp(dev_preds)-1                 
print(np.mean(dev_preds))
print("*"*80)

####################################################################################

#测试模型
print(test.shape)
print(train.shape)
scaler=preprocessing.MinMaxScaler(feature_range=(0,8))
scaler.fit(train[['imp']]) 
hparam.train_scaler=scaler
hparam.test_scaler=scaler
index=set(range(train.shape[0]))
K_fold=[]
for i in range(5):
    if i == 4:
        tmp=index
    else:
        tmp=random.sample(index,int(1.0/5*train.shape[0]))
    index=index-set(tmp)
    print("Number:",len(tmp))
    K_fold.append(tmp)

train_preds=np.zeros(len(train))
test_preds=np.zeros(len(test))
scores=[]
train['gold']=True
for i in range(5):
    print("Fold",i)
    dev_index=K_fold[i]
    train_index=[]
    for j in range(5):
        if j!=i:
            train_index+=K_fold[j]
    for k in range(2):
        model=ctrNet.build_model(hparam)
        score=model.train(train.loc[train_index],train.loc[dev_index])
        scores.append(score)
        train_preds[list(dev_index)]+=model.infer(train.loc[list(dev_index)])/2
        test_preds+=model.infer(test)/10
        print(np.mean((np.exp(test_preds*10/(i*2+k+1))-1)))
    try:
        del model
        gc.collect()
    except:
        pass
train_preds=np.exp(train_preds)-1
test_preds=np.exp(test_preds)-1

####################################################################################

#输出
print(scores)
print(np.mean(scores))    
print(train_preds.mean())
print(dev_preds.mean())
print(test_preds.mean())
train_fea = train[['aid','request_day']]
train_fea['nn_preds']=train_preds
dev['nn_preds'] = dev_preds
dev_fea=dev[['aid','bid','gold','imp','nn_preds']]
test['nn_preds'] = test_preds
test_fea=test[['aid','nn_preds']]
train_fea.to_csv('submission/nn_pred_{}_train.csv'.format(hparam.model_name),index=False)
test_fea.to_csv('submission/nn_pred_{}_test.csv'.format(hparam.model_name),index=False)
dev_fea.to_csv('submission/nn_pred_{}_dev.csv'.format(hparam.model_name),index=False)
####################################################################################