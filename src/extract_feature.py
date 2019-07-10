import os
import pandas as pd
import numpy as np
import random
import json
import gc
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from collections import Counter
from sklearn import preprocessing
import scipy.special as special
from pandas import DataFrame, Series
from tqdm import tqdm
import time
np.random.seed(2019)
random.seed(2019)


def w2v(log,pivot,f,flag,L):
    #word2vec算法
    #log为曝光日志，以pivot为主键，f为embedding的对象，flag为dev或test，L是embedding的维度
    print("w2v:",pivot,f)
    
    #构造文档
    log[f]=log[f].fillna(-1).astype(int)
    sentence=[]
    dic={}
    day=0
    log=log.sort_values(by='request_day')
    log['day']=log['request_day']
    for item in log[['day',pivot,f]].values:
        if day!=item[0]:
            for key in dic:
                sentence.append(dic[key])
            dic={}
            day=item[0]
        try:
            dic[item[1]].append(str(int(item[2])))
        except:
            dic[item[1]]=[str(int(item[2]))]
    for key in dic:
        sentence.append(dic[key])
    print(len(sentence))
    #训练Word2Vec模型
    print('training...')
    random.shuffle(sentence)
    model = Word2Vec(sentence, size=L, window=10, min_count=1, workers=10,iter=10)
    print('outputing...')
    #保存文件
    values=set(log[f].values)
    w2v=[]
    for v in values:
        try:
            a=[int(v)]
            a.extend(model[str(v)])
            w2v.append(a)
        except:
            pass
    out_df=pd.DataFrame(w2v)
    names=[f]
    for i in range(L):
        names.append(pivot+'_w2v_embedding_'+f+'_'+str(L)+'_'+str(i))
    out_df.columns = names
    out_df.to_pickle('data/' +pivot+'_'+ f +'_'+flag +'_w2v_'+str(L)+'.pkl') 

def deepwalk(log,f1,f2,flag,L):
    #Deepwalk算法，
    print("deepwalk:",f1,f2)
    #构建图
    dic={}
    for item in log[[f1,f2]].values:
        try:
            str(int(item[1]))
            str(int(item[0]))
        except:
            continue
        try:
            dic['item_'+str(int(item[1]))].add('user_'+str(int(item[0])))
        except:
            dic['item_'+str(int(item[1]))]=set(['user_'+str(int(item[0]))])
        try:
            dic['user_'+str(int(item[0]))].add('item_'+str(int(item[1])))
        except:
            dic['user_'+str(int(item[0]))]=set(['item_'+str(int(item[1]))])
    dic_cont={}
    for key in dic:
        dic[key]=list(dic[key])
        dic_cont[key]=len(dic[key])
    print("creating")     
    #构建路径
    path_length=10        
    sentences=[]
    length=[]
    for key in dic:
        sentence=[key]
        while len(sentence)!=path_length:
            key=dic[sentence[-1]][random.randint(0,dic_cont[sentence[-1]]-1)]
            if len(sentence)>=2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)
        sentences.append(sentence)
        length.append(len(sentence))
        if len(sentences)%100000==0:
            print(len(sentences))
    print(np.mean(length))
    print(len(sentences))
    #训练Deepwalk模型
    print('training...')
    random.shuffle(sentences)
    model = Word2Vec(sentences, size=L, window=4,min_count=1,sg=1, workers=10,iter=20)
    print('outputing...')
    #输出
    values=set(log[f1].values)
    w2v=[]
    for v in values:
        try:
            a=[int(v)]
            a.extend(model['user_'+str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df=pd.DataFrame(w2v)
    names=[f1]
    for i in range(L):
        names.append(f1+'_'+ f2+'_'+names[0]+'_deepwalk_embedding_'+str(L)+'_'+str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle('data/' +f1+'_'+ f2+'_'+f1 +'_'+flag +'_deepwalk_'+str(L)+'.pkl') 
    ########################
    values=set(log[f2].values)
    w2v=[]
    for v in values:
        try:
            a=[int(v)]
            a.extend(model['item_'+str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df=pd.DataFrame(w2v)
    names=[f2]
    for i in range(L):
        names.append(f1+'_'+ f2+'_'+names[0]+'_deepwalk_embedding_'+str(L)+'_'+str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle('data/' +f1+'_'+ f2+'_'+f2 +'_'+flag +'_deepwalk_'+str(L)+'.pkl') 
    
    
    

def predict_periods(train_df,test_df,wday):
    #提取测试当天投放时段特征
    print("predict_periods features")    
    #提取训练集的投放时段特征，分别有48维的01向量，和投放时段总数
    items=[]
    for item in train_df[['wday','delivery_periods']].values:
        w=item[0]
        item=item[1]
        val=int(item.split(',')[w])
        temp=[]
        for i in range(48):
            if val%2==1:
                temp.append(1)
            else:
                temp.append(0)
            val//=2
        assert val==0
        temp.append(sum(temp))
        items.append(temp)
    df=pd.DataFrame(items)
    df.columns=['periods_on_'+str(i) for i in range(48)]+['periods_cont']
    for f in ['periods_on_'+str(i) for i in range(48)]+['periods_cont']:
        train_df[f]=df[f]
    del df
    del items
    gc.collect()    
    #提取测试集的投放时段特征，分别有48维的01向量，和投放时段总数
    items=[]
    for item in test_df['delivery_periods'].values:
        val=int(item.split(',')[wday])
        temp=[]
        for i in range(48):
            if val%2==1:
                temp.append(1)
            else:
                temp.append(0)
            val//=2
        assert val==0
        temp.append(sum(temp))
        items.append(temp)
    df=pd.DataFrame(items)
    df.columns=['periods_on_'+str(i) for i in range(48)]+['periods_cont']
    for f in ['periods_on_'+str(i) for i in range(48)]+['periods_cont']:
        test_df[f]=df[f]
    del df
    del items
    gc.collect()

    
def crowd_uid(train_df,test_df,f1,f2,log,k):
    #多值特征，提取以f1为主键，f2在log中出现Topk的ID
    #如f1=aid,f2=uid，k=100,则表示访问该广告最多的前100名用户
    print("crowd_uid features",f1,f2)
    dic={}
    log[f1]=log[f1].fillna(-1).astype(int)
    train_df[f1]=train_df[f1].fillna(-1).astype(int)
    test_df[f1]=test_df[f1].fillna(-1).astype(int)
    for item in tqdm(log[[f1,f2,'request_day']].values,total=len(log)):
        try:
            dic[item[0]][0][item[1]]+=1
        except:
            dic[item[0]]=[Counter(),Counter()]
            dic[item[0]][0][item[1]]=1
            
    items=[]
    for key in tqdm(dic,total=len(dic)):
        conter=dic[key][0]
        item=[str(x[0]) for x in conter.most_common(k)]
        if len(item)==0:
            item=['-1']
        items.append([key,' '.join(item)])
    
    df=pd.DataFrame(items)
    df.columns=[f1,f1+'_'+f2+'s']
    df = df.drop_duplicates(f1)
    try:
        del train_df[f1+'_'+f2+'s']
        del test_df[f1+'_'+f2+'s']
    except:
        pass
    train_df=train_df.merge(df,on=f1,how='left')
    test_df=test_df.merge(df,on=f1,how='left')
    train_df[f1+'_'+f2+'s']=train_df[f1+'_'+f2+'s'].fillna('-1')
    test_df[f1+'_'+f2+'s']=test_df[f1+'_'+f2+'s'].fillna('-1')
    del df
    del items
    del dic
    gc.collect()
    return train_df,test_df  




def crowd_direction(train_df,test_df):
    #人群定向，多值特征
    print("crowd_direction features")
    index={
       'age':0,
        'gender':1,
        'area':2,
        'status':3,
        'education':4,
        'consuptionAbility':5,
        'device':6,
        'os':6,
        'work':7,
        'connectionType':8,
        'behavior':9
    }
    revert_index={}
    for key in index:
        revert_index[index[key]]=key    
    #构造训练集的人群定向，有年龄，性别等多值特征
    items=[]
    for item in train_df['crowd_direction'].values:
        if item=='all':
            items.append(['all']*10)
        else:
            temp=['all' for i in range(10)]
            for features in item.split('|'):
                key=features.split(':')[0]
                val=features.split(':')[1].split(',')
                temp[index[key]]=' '.join(val)
            items.append(temp)
    df=pd.DataFrame(items)
    df.columns=['age','gender','area','status','education','consuptionAbility','os','work','connectionType','behavior']
    for f in ['age','gender','area','status','education','consuptionAbility','os','work','connectionType','behavior']:
        train_df[f]=df[f]
    del df
    del items
    gc.collect() 
    #构造测试集的人群定向，有年龄，性别等多值特征
    items=[]
    for item in test_df['crowd_direction'].values:
        if item=='all':
            items.append(['all']*10)
        else:
            temp=['all' for i in range(10)]
            for features in item.split('|'):
                key=features.split(':')[0]
                val=features.split(':')[1].split(',')
                temp[index[key]]=' '.join(val)
            items.append(temp)
    df=pd.DataFrame(items)
    df.columns=['age','gender','area','status','education','consuptionAbility','os','work','connectionType','behavior']
    for f in ['age','gender','area','status','education','consuptionAbility','os','work','connectionType','behavior']:
        test_df[f]=df[f]
    del df
    del items
    gc.collect()   
                    

def history(train_df,test_df,log,pivot,f):
    #以pivot为主键，统计最近一次f的值
    print("history",pivot,f)
    nan=log[f].median()
    dic={}
    temp_log=log[[pivot,'request_day',f,'aid']].drop_duplicates(['aid','request_day'],keep='last')
    for item in log[[pivot,'request_day',f]].values:
        if (item[0],item[1]) not in dic:
            dic[(item[0],item[1])]=[item[2]]
        else:
            dic[(item[0],item[1])].append(item[2])
    for key in dic:
        dic[key]=np.mean(dic[key])
    #统计训练集的特征
    items=[]
    cont=0
    day=log['request_day'].min()
    for item in train_df[[pivot,'request_day']].values:
        flag=False
        for i in range(item[1]-1,day-1,-1):
            if (item[0],i) in dic:
                items.append(dic[(item[0],i)])
                flag=True
                cont+=1
                break
        if flag is False:
            items.append(nan)
    train_df['history_'+pivot+'_'+f]=items
    #统计测试集的特征
    items=[]
    cont=0
    day_min=log['request_day'].min()
    day_max=log['request_day'].max()
    for item in test_df[pivot].values:
        flag=False
        for i in range(day_max,day_min-1,-1):
            if (item,i) in dic:
                items.append(dic[(item,i)])
                flag=True
                cont+=1
                break
        if flag is False:
            items.append(nan)
    test_df['history_'+pivot+'_'+f]=items
    
    print(train_df['history_'+pivot+'_'+f].mean())
    print(test_df['history_'+pivot+'_'+f].mean())
    del items
    del dic
    gc.collect()
    return train_df,test_df

def get_agg_features(train_df,test_df,f1,f2,agg,log=None):
    if type(f1)==str:
        f1=[f1]
    if log is None:
        if agg!='size':
            data=train_df[f1+[f2]].append(test_df.drop_duplicates(f1+[f2])[f1+[f2]])
        else:
            data=train_df[f1].append(test_df.drop_duplicates(f1)[f1])
    else:
        if agg!='size':
            data=log[f1+[f2]]
        else:
            data=log[f1]
    if agg=="size":
        tmp = pd.DataFrame(data.groupby(f1).size()).reset_index()
    elif agg == "count":
        tmp = pd.DataFrame(data.groupby(f1)[f2].count()).reset_index()
    elif agg=="mean":
        tmp = pd.DataFrame(data.groupby(f1)[f2].mean()).reset_index()
    elif agg=="unique":
        tmp = pd.DataFrame(data.groupby(f1)[f2].nunique()).reset_index()
    elif agg=="max":
        tmp = pd.DataFrame(data.groupby(f1)[f2].max()).reset_index()
    elif agg=="min":
        tmp = pd.DataFrame(data.groupby(f1)[f2].min()).reset_index()
    elif agg=="sum":
        tmp = pd.DataFrame(data.groupby(f1)[f2].sum()).reset_index()
    elif agg=="std":
        tmp = pd.DataFrame(data.groupby(f1)[f2].std()).reset_index()
    elif agg=="median":
        tmp = pd.DataFrame(data.groupby(f1)[f2].median()).reset_index()
    elif agg=="skew":
        tmp = pd.DataFrame(data.groupby(f1)[f2].skew()).reset_index()
    elif agg=="unique_mean":
        group=data.groupby(f1)
        group=group.apply(lambda x:np.mean(list(Counter(list(x[f2])).values())))
        tmp = pd.DataFrame(group.reset_index())
    elif agg=="unique_var":
        group=data.groupby(f1)
        group=group.apply(lambda x:np.var(list(Counter(list(x[f2])).values())))
        tmp = pd.DataFrame(group.reset_index())
    else:
        raise "agg error"
    if log is None:
        tmp.columns = f1+['_'.join(f1)+"_"+f2+"_"+agg]
        print('_'.join(f1)+"_"+f2+"_"+agg)
    else:
        tmp.columns = f1+['_'.join(f1)+"_"+f2+"_log_"+agg]
        print('_'.join(f1)+"_"+f2+"_log_"+agg)
    try:
        del test_df['_'.join(f1)+"_"+f2+"_"+agg]
        del train_df['_'.join(f1)+"_"+f2+"_"+agg]
    except:
        pass
    test_df=test_df.merge(tmp, on=f1, how='left')
    train_df=train_df.merge(tmp, on=f1, how='left')
    del tmp
    del data
    gc.collect()
    print(train_df.shape,test_df.shape)
    return train_df,test_df  

def kfold_static(train_df,test_df,f,label):
    print("K-fold static:",f+'_'+label)
    #K-fold positive and negative num
    avg_rate=train_df[label].mean()
    num=len(train_df)//5
    index=[0 for i in range(num)]+[1 for i in range(num)]+[2 for i in range(num)]+[3 for i in range(num)]+[4 for i in range(len(train_df)-4*num)]
    random.shuffle(index)
    train_df['index']=index
    #五折统计
    dic=[{} for i in range(5)]
    dic_all={}
    for item in train_df[['index',f,label]].values:
        try:
            dic[item[0]][item[1]].append(item[2])
        except:
            dic[item[0]][item[1]]=[]
            dic[item[0]][item[1]].append(item[2])
    print("static done!")
    #构造训练集的五折特征，均值，中位数等       
    mean=[]
    median=[]
    std=[]
    Min=[]
    Max=[]
    cache={}
    for item in train_df[['index',f]].values:
        if tuple(item) not in cache:
            temp=[]
            for i in range(5):
                 if i!=item[0]:
                    try:
                        temp+=dic[i][item[1]]
                    except:
                        pass
            if len(temp)==0:
                cache[tuple(item)]=[-1]*5
            else:
                cache[tuple(item)]=[np.mean(temp),np.median(temp),np.std(temp),np.min(temp),np.max(temp)]
        temp=cache[tuple(item)]
        mean.append(temp[0])
        median.append(temp[1])
        std.append(temp[2])
        Min.append(temp[3])
        Max.append(temp[4])                     
    del cache        
    train_df[f+'_'+label+'_mean']=mean
    train_df[f+'_'+label+'_median']=median
    train_df[f+'_'+label+'_std']=std
    train_df[f+'_'+label+'_min']=Min
    train_df[f+'_'+label+'_max']=Max   
    print("train done!")
    
    #构造测试集的五折特征，均值，中位数等  
    mean=[]
    median=[]
    std=[]
    Min=[]
    Max=[]
    cache={}
    for uid in test_df[f].values:
        if uid not in cache:
            temp=[]
            for i in range(5):
                try:
                    temp+=dic[i][uid]
                except:
                    pass
            if len(temp)==0:
                cache[uid]=[-1]*5
            else:
                cache[uid]=[np.mean(temp),np.median(temp),np.std(temp),np.min(temp),np.max(temp)]
        temp=cache[uid]
        mean.append(temp[0])
        median.append(temp[1])
        std.append(temp[2])
        Min.append(temp[3])
        Max.append(temp[4])           
        
    test_df[f+'_'+label+'_mean']=mean
    test_df[f+'_'+label+'_median']=median
    test_df[f+'_'+label+'_std']=std
    test_df[f+'_'+label+'_min']=Min
    test_df[f+'_'+label+'_max']=Max   
    print("test done!")
    del train_df['index']
    print(f+'_'+label+'_mean')
    print(f+'_'+label+'_median')
    print(f+'_'+label+'_std')
    print(f+'_'+label+'_min')
    print(f+'_'+label+'_max')
    print('avg of mean',np.mean(train_df[f+'_'+label+'_mean']),np.mean(test_df[f+'_'+label+'_mean']))
    print('avg of median',np.mean(train_df[f+'_'+label+'_median']),np.mean(test_df[f+'_'+label+'_median']))
    print('avg of std',np.mean(train_df[f+'_'+label+'_std']),np.mean(test_df[f+'_'+label+'_std']))
    print('avg of min',np.mean(train_df[f+'_'+label+'_min']),np.mean(test_df[f+'_'+label+'_min']))
    print('avg of max',np.mean(train_df[f+'_'+label+'_max']),np.mean(test_df[f+'_'+label+'_max']))
    
    
    
    
    
if __name__ == "__main__":    
    for path1,path2,log_path,flag,wday,day in [('data/train_dev.pkl','data/dev.pkl','data/user_log_dev.pkl','dev',1,17974),('data/train.pkl','data/test.pkl','data/user_log_test.pkl','test',3,17976)]:
            ##拼接静态特征
            print(path1,path2,log_path,flag)
            train_df=pd.read_pickle(path1)
            test_df=pd.read_pickle(path2)
            log=pd.read_pickle(log_path)
            print(train_df.shape,test_df.shape,log.shape)
            df =pd.read_pickle('data/testA/ad_static_feature.pkl')
            log=log.merge(df,on='aid',how='left')
            del df
            gc.collect()
            print(train_df.shape,test_df.shape,log.shape)
       
            
            #提取特征
            
            #人群定向
            crowd_direction(train_df,test_df)
            #投放时段
            predict_periods(train_df,test_df,wday)
            #多值特征
            train_df,test_df=crowd_uid(train_df,test_df,'good_id','advertiser',log,100)
            train_df,test_df=crowd_uid(train_df,test_df,'good_id','request_day',log,100)
            train_df,test_df=crowd_uid(train_df,test_df,'good_id','position',log,100)
            train_df,test_df=crowd_uid(train_df,test_df,'good_id','period_id',log,100)
            train_df,test_df=crowd_uid(train_df,test_df,'good_id','wday',log,100)
            train_df,test_df=crowd_uid(train_df,test_df,'advertiser','good_id',log,100)
            train_df,test_df=crowd_uid(train_df,test_df,'advertiser','request_day',log,100)
            train_df,test_df=crowd_uid(train_df,test_df,'advertiser','position',log,100)
            train_df,test_df=crowd_uid(train_df,test_df,'advertiser','period_id',log,100)
            train_df,test_df=crowd_uid(train_df,test_df,'advertiser','wday',log,100)
            train_df,test_df=crowd_uid(train_df,test_df,'aid','uid',log,20)
            #历史特征
            for pivot in ['aid']:
                for f in ['imp','bid','pctr','quality_ecpm','totalEcpm']:
                    history(train_df,test_df,log,pivot,f)  
            #五折特征
            kfold_static(train_df,test_df,'aid','imp')
            kfold_static(train_df,test_df,'good_id','imp')
            kfold_static(train_df,test_df,'advertiser','imp')
            #统计特征
            train_df,test_df=get_agg_features(train_df,test_df,["good_id"],'advertiser',"count")
            train_df,test_df=get_agg_features(train_df,test_df,["good_id"],'aid',"count") 
            train_df,test_df=get_agg_features(train_df,test_df,["good_id"],'ad_size',"count") 
            train_df,test_df=get_agg_features(train_df,test_df,["good_id"],'ad_type_id',"count") 
            train_df,test_df=get_agg_features(train_df,test_df,["good_id"],'good_id',"size")
            train_df,test_df=get_agg_features(train_df,test_df,["advertiser"],'good_id',"count") 
            train_df,test_df=get_agg_features(train_df,test_df,["advertiser"],'aid',"count") 
            train_df,test_df=get_agg_features(train_df,test_df,["advertiser"],'ad_size',"count")    
            train_df,test_df=get_agg_features(train_df,test_df,["advertiser"],'ad_type_id',"count")     
            train_df,test_df=get_agg_features(train_df,test_df,["advertiser"],'good_type',"count") 
            train_df,test_df=get_agg_features(train_df,test_df,["advertiser"],'advertiser',"size") 
            train_df,test_df=get_agg_features(train_df,test_df,['good_type'],'good_type',"size")                                  
            train_df,test_df=get_agg_features(train_df,test_df,["aid"],'aid',"size") 
            #保存数据
            print(train_df.shape,test_df.shape,log.shape)
            train_df.to_pickle(path1) 
            test_df.to_pickle(path2)  
            print(list(train_df))
            print("*"*80)
            print("save done!")
            
            #Word2vec
            w2v(log,'uid','good_id',flag,64)
            w2v(log,'uid','advertiser',flag,64)
            w2v(log,'uid','aid',flag,64)
            #Deepwalk
            deepwalk(log,'uid','aid',flag,64)
            deepwalk(log,'uid','good_id',flag,64)

            del train_df
            del test_df
            del log
            gc.collect()
