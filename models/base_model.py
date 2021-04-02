"""define base class model"""
import abc
import math
import tensorflow as tf
from sklearn import metrics
import os
from src import misc_utils as utils
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import time
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing
__all__ = ["BaseModel"]


class BaseModel(object):
    def __init__(self, hparams,  scope=None):
        tf.set_random_seed(1234)
       
        
    @abc.abstractmethod
    def _build_graph(self, hparams):
        """Subclass must implement this."""
        pass


    def _get_initializer(self, hparams):
        if hparams.init_method == 'tnormal':
            return tf.truncated_normal_initializer(stddev=hparams.init_value)
        elif hparams.init_method == 'uniform':
            return tf.random_uniform_initializer(-hparams.init_value, hparams.init_value)
        elif hparams.init_method == 'normal':
            return tf.random_normal_initializer(stddev=hparams.init_value)
        elif hparams.init_method == 'xavier_normal':
            return tf.contrib.layers.xavier_initializer(uniform=False)
        elif hparams.init_method == 'xavier_uniform':
            return tf.contrib.layers.xavier_initializer(uniform=True)
        elif hparams.init_method == 'he_normal':
            return tf.contrib.layers.variance_scaling_initializer( \
                factor=2.0, mode='FAN_AVG', uniform=False)
        elif hparams.init_method == 'he_uniform':
            return tf.contrib.layers.variance_scaling_initializer( \
                factor=2.0, mode='FAN_AVG', uniform=True)
        else:
            return tf.truncated_normal_initializer(stddev=hparams.init_value)


    def _build_train_opt(self, hparams):
        def train_opt(hparams):
            if hparams.optimizer == 'adadelta':
                train_step = tf.train.AdadeltaOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'adagrad':
                train_step = tf.train.AdagradOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'sgd':
                train_step = tf.train.GradientDescentOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'adam':
                train_step = tf.train.AdamOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'ftrl':
                train_step = tf.train.FtrlOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'gd':
                train_step = tf.train.GradientDescentOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'padagrad':
                train_step = tf.train.ProximalAdagradOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'pgd':
                train_step = tf.train.ProximalGradientDescentOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'rmsprop':
                train_step = tf.train.RMSPropOptimizer( \
                    hparams.learning_rate)
            else:
                train_step = tf.train.GradientDescentOptimizer( \
                    hparams.learning_rate)
            return train_step

        train_step = train_opt(hparams)
        return train_step
    
        
        
    def _active_layer(self, logit, scope, activation, layer_idx):
        logit = self._activate(logit, activation)
        return logit

    def _activate(self, logit, activation):
        if activation == 'sigmoid':
            return tf.nn.sigmoid(logit)
        elif activation == 'softmax':
            return tf.nn.softmax(logit)
        elif activation == 'relu':
            return tf.nn.relu(logit)
        elif activation == 'tanh':
            return tf.nn.tanh(logit)
        elif activation == 'elu':
            return tf.nn.elu(logit)
        elif activation == 'identity':
            return tf.identity(logit)
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _dropout(self, logit, layer_idx):
        logit = tf.nn.dropout(x=logit, keep_prob=self.layer_keeps[layer_idx])
        return logit


    
    def batch_norm_layer(self, x, train_phase, scope_bn):
        z = tf.cond(train_phase, lambda: batch_norm(x, decay=self.hparams.batch_norm_decay, center=True, scale=True, updates_collections=None,is_training=True, reuse=None, trainable=True, scope=scope_bn), lambda: batch_norm(x, decay=self.hparams.batch_norm_decay, center=True, scale=True, updates_collections=None,is_training=False, reuse=True, trainable=False, scope=scope_bn))
        return z
    
    def optimizer(self,hparams):
        opt=self._build_train_opt(hparams)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss,params,colocate_gradients_with_ops=True)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)  
        self.grad_norm =gradient_norm 
        self.update = opt.apply_gradients(zip(clipped_grads, params)) 

    def train(self,train,dev):
        hparams=self.hparams
        sess=self.sess
        train_single_features=train[hparams.single_features].values
        train_label=train[[hparams.label]].values
        if hparams.multi_features is not None:
            train_multi_features=train[hparams.multi_features].values
        if hparams.dense_features is not None:
            train_dense_features=train[hparams.dense_features].values
        if hparams.kv_features is not None:
            train_kv_features=train[hparams.kv_features].values
 
        if hparams.cross_features is not None:
            train_cross_features=train[hparams.cross_features].values
        for epoch in range(hparams.epoch):
            info={}
            info['loss']=[]
            info['norm']=[]
            start_time = time.time()
            for idx in range(len(train)//hparams.batch_size+3):
                if idx*hparams.batch_size>=len(train):
                    T=(time.time()-start_time)
                    break
                feed_dic={} 
                if hparams.single_features is not None:
                    single_batch=train_single_features[idx*hparams.batch_size:min((idx+1)*hparams.batch_size,len(train))]
                    single_batch=utils.hash_single_batch(single_batch,hparams)
                    feed_dic[self.single_features]=single_batch
                
                if hparams.multi_features is not None:
                    multi_batch=train_multi_features[idx*hparams.batch_size:min((idx+1)*hparams.batch_size,len(train))]     
                    multi_batch,multi_weights=utils.hash_multi_batch(multi_batch,hparams)
                    feed_dic[self.multi_features]=multi_batch 
                    feed_dic[self.multi_weights]=multi_weights
                    
                if hparams.dense_features is not None:
                    feed_dic[self.dense_features]=train_dense_features[idx*hparams.batch_size:\
                                                                       min((idx+1)*hparams.batch_size,len(train))]
                if hparams.kv_features is not None:
                    feed_dic[self.kv_features]=train_kv_features[idx*hparams.batch_size:\
                                                                       min((idx+1)*hparams.batch_size,len(train))]                                      

                if hparams.cross_features is not None:
                    cross_batch=train_cross_features[idx*hparams.batch_size:min((idx+1)*hparams.batch_size,len(train))]
                    cross_batch=utils.hash_single_batch(cross_batch,hparams)
                    feed_dic[self.cross_features]=cross_batch                            
                label=train_label[idx*hparams.batch_size: min((idx+1)*hparams.batch_size,len(train))]
                label=hparams.train_scaler.transform(label)[:,0]
                feed_dic[self.label]=label
                feed_dic[self.use_norm]=True
                loss,_,norm=sess.run([self.score,self.update,self.grad_norm],feed_dict=feed_dic)

                info['loss'].append(loss)
                info['norm'].append(norm)
                if (idx+1)%hparams.num_display_steps==0:                   
                    info['learning_rate']=hparams.learning_rate
                    info["train_ppl"]= np.mean(info['loss'])
                    info["avg_grad_norm"]=np.mean(info['norm'])
                    utils.print_step_info("  ", epoch,idx+1, info)
                    del info
                    info={}
                    info['loss']=[]
                    info['norm']=[]
                if (idx+1)%hparams.num_eval_steps==0:
                    T=(time.time()-start_time)
                    if dev is not None:
                        self.eval(T,dev,hparams,sess)

        if dev is not None:
            return self.eval(T,dev,hparams,sess)
        else:
            return 
    
    def reload(self):
        hparams=self.hparams
        self.saver.restore(self.sess,'model_tmp/model/'+hparams.model_name)

    def save(self):
        hparams = self.hparams
        self.saver.save(self.sess, 'model_tmp/model/' + hparams.model_name)


    def infer(self,dev):
        hparams=self.hparams
        sess=self.sess   
        preds=[]
        total_loss=[]
        a=hparams.batch_size
        hparams.batch_size=hparams.infer_batch_size
        dev_single_features=dev[hparams.single_features].values
        if hparams.multi_features is not None:
            dev_multi_features=dev[hparams.multi_features].values 
        if hparams.dense_features is not None:
            dev_dense_features=dev[hparams.dense_features].values 
        if hparams.kv_features is not None:
            dev_kv_features=dev[hparams.kv_features].values 

        if hparams.cross_features is not None:
            dev_cross_features=dev[hparams.cross_features].values
                                      
        for idx in tqdm(range(len(dev)//hparams.batch_size+1),total=len(dev)//hparams.batch_size+1):
            single_batch=dev_single_features[idx*hparams.batch_size:min((idx+1)*hparams.batch_size,len(dev))]
            if len(single_batch)==0:
                break
            feed_dic={}
            feed_dic[self.use_norm]=False
            
            if hparams.single_features is not None:    
                single_batch=utils.hash_single_batch(single_batch,hparams)
                feed_dic[self.single_features]=single_batch

            if hparams.multi_features is not None:
                multi_batch=dev_multi_features[idx*hparams.batch_size:min((idx+1)*hparams.batch_size,len(dev))]              
                multi_batch,multi_weights=utils.hash_multi_batch(multi_batch,hparams)
                feed_dic[self.multi_features]=multi_batch 
                feed_dic[self.multi_weights]=multi_weights        
            if hparams.dense_features is not None:
                feed_dic[self.dense_features]=dev_dense_features[idx*hparams.batch_size:\
                                                                       min((idx+1)*hparams.batch_size,len(dev))]  
            if hparams.kv_features is not None:
                feed_dic[self.kv_features]=dev_kv_features[idx*hparams.batch_size:\
                                                                       min((idx+1)*hparams.batch_size,len(dev))] 

            if hparams.cross_features is not None:
                cross_batch=dev_cross_features[idx*hparams.batch_size:min((idx+1)*hparams.batch_size,len(dev))]
                cross_batch=utils.hash_single_batch(cross_batch,hparams)
                feed_dic[self.cross_features]=cross_batch 
            feed_dic[self.use_norm]=False
            pred=sess.run(self.val,feed_dict=feed_dic)  
            preds.append(pred)   
        preds=np.concatenate(preds)
        dev['temp']=preds
        preds=hparams.test_scaler.inverse_transform(dev[['temp']])[:,0]
        del dev['temp']
        hparams.batch_size=a
        return preds

                        

    def eval(self,T,dev,hparams,sess):
        preds=self.infer(dev)
        dev['predict_imp']=preds
        
        dev['rank']=dev[['aid', 'bid']].groupby('aid')['bid'].apply(lambda row: pd.Series(dict(zip(row.index, row.rank()))))-1
        dev['predict_imp']=dev['predict_imp'].apply(lambda x:np.exp(x)-1)
        dev['predict_imp']=dev['predict_imp'].apply(round)
        dev['predict_imp']=dev['predict_imp'].apply(lambda x: 0 if x<0  else x)
        dev['predict_imp']=dev['predict_imp']+dev['rank']*0.0001
        
        dev['predict_imp']=dev['predict_imp'].apply(lambda x: round(x,4))
        gold_dev=dev[dev['gold']==True]
        score=abs(gold_dev['gold_imp']-gold_dev['predict_imp'])/((gold_dev['gold_imp']+gold_dev['predict_imp'])/2+1e-15)
        SMAPE=score.mean()
        
        try:
            last_aid=None
            gold_imp=None
            gold_bid=None
            s=None
            score=[]
            for item in dev[['aid','bid','predict_imp']].values:
                item=list(item)
                if item[0]!=last_aid:
                    last_aid=item[0]
                    gold_bid=item[1]
                    gold_imp=item[2]
                    if s is not None:
                        score.append(s/cont)
                    s=0
                    cont=0
                else:
                    if (gold_imp-item[2])*(gold_bid-item[1])==0:
                        s+=-1
                    else:
                        s+=((gold_imp-item[2])*(gold_bid-item[1]))/(abs(((gold_imp-item[2])*(gold_bid-item[1]))))
                    cont+=1

            MonoScore=np.mean(score)        
            score=0.4*(1-SMAPE/2)+0.6*(MonoScore+1)/2
        except:
            MonoScore=0

        if SMAPE<self.best_score:
            self.best_score=SMAPE
        utils.print_out(("# Epcho-time %.2fs AVG %.4f. Eval SMAPE %.4f. #Eval MonoScore %.4f. Best Score %.4f")%(T,dev['predict_imp'].mean(),SMAPE,MonoScore,self.best_score)) 
        return SMAPE


