import tensorflow as tf
from src import misc_utils as utils
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core
from models.base_model import BaseModel
import numpy as np
import time 
import os
class Model(BaseModel):
    def __init__(self,hparams):
        self.hparams=hparams
        if hparams.metric in ['SMAPE']:
            self.best_score=100000
        else:
            self.best_score=0
        self.build_graph(hparams)   
        self.optimizer(hparams)

              
        params = tf.trainable_variables()
        utils.print_out("# Trainable variables")
        for param in params:
            utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),param.op.device))   
  
    def set_Session(self,sess):
        self.sess=sess
        
    def build_graph(self, hparams):
        self.initializer = self._get_initializer(hparams)
        self.label = tf.placeholder(shape=(None), dtype=tf.float32)
        self.use_norm=tf.placeholder(tf.bool)
        hparams.feature_nums=len(hparams.cross_features)
        emb_inp_v2=[]
        dnn_input=[]
        
        
        if hparams.dense_features is not None:
            self.dense_features=tf.placeholder(shape=(None,len(hparams.dense_features)), dtype=tf.float32)
            
        if hparams.kv_features is not None:
            self.kv_features=tf.placeholder(shape=(None,len(hparams.kv_features)), dtype=tf.float32)
            kv_emb_v2=tf.get_variable(shape=[len(hparams.kv_features),hparams.kv_batch_num+1,hparams.k],
                                                  initializer=self.initializer,name='emb_v2_kv')
            
        if hparams.multi_features is not None:
            hparams.feature_nums+=len(hparams.multi_features)
            
        #CIN: 结合多值特征和交叉特征输入CIN中 
        if hparams.cross_features is not None:    
            self.cross_features=tf.placeholder(shape=(None,len(hparams.cross_features)), dtype=tf.int32)
            self.cross_emb_v2=tf.get_variable(shape=[hparams.cross_hash_num,hparams.k],initializer=self.initializer,name='emb_v2_cross')
            emb_inp_v2.append(tf.gather(self.cross_emb_v2, self.cross_features))
        
        if hparams.multi_features is not None:
            self.multi_features=tf.placeholder(shape=(None,len(hparams.multi_features),None), dtype=tf.int32)
            self.multi_weights=tf.placeholder(shape=(None,len(hparams.multi_features),None), dtype=tf.float32)
            self.multi_emb_v2=tf.get_variable(shape=[hparams.multi_hash_num,hparams.k],
                                                     initializer=self.initializer,name='emb_v2_multi') 
            emb_multi_v2=tf.gather(self.multi_emb_v2, self.multi_features)
            self.weights=self.multi_weights/(tf.reduce_sum(self.multi_weights,-1)+1e-20)[:,:,None]
            emb_multi_v2=tf.reduce_sum(emb_multi_v2*self.weights[:,:,:,None],2)
            emb_inp_v2.append(emb_multi_v2)
            
        if len(emb_inp_v2)!=0:
            emb_inp_v2=tf.concat(emb_inp_v2,1)
            result=self._build_extreme_FM(hparams, emb_inp_v2, res=False, direct=False, bias=False, reduce_D=False, f_dim=2)
            dnn_input.append(tf.reshape(emb_inp_v2,[-1,hparams.feature_nums*hparams.k]))
            dnn_input.append(result)
        
        #单值特征，直接embedding
        if hparams.single_features is not None:
            self.single_features=tf.placeholder(shape=(None,len(hparams.single_features)), dtype=tf.int32)
            self.single_emb_v2=tf.get_variable(shape=[hparams.single_hash_num,hparams.single_k],
                                               initializer=self.initializer,name='emb_v2_single') 
            dnn_input.append(tf.reshape(tf.gather(self.single_emb_v2, self.single_features),[-1,len(hparams.single_features)*hparams.single_k]))
        
        #key-values memory
        if hparams.kv_features is not None:
            index=[i/hparams.kv_batch_num for i in range(hparams.kv_batch_num+1)]    
            index=tf.constant(index)  
            distance=1/(tf.abs(self.kv_features[:,:,None]-index[None,None,:])+0.00001)
            weights=tf.nn.softmax(distance,-1) #[batch_size,kv_features_size,kv_batch_num]
            kv_emb=tf.reduce_sum(weights[:,:,:,None]*kv_emb_v2[None,:,:,:],-2)
            kv_emb=tf.reshape(kv_emb,[-1,len(hparams.kv_features)*hparams.k])
            dnn_input.append(kv_emb)
        
        #稠密特征，这里主要放embedding特征，即word2vec和deepwalk
        if hparams.dense_features is not None:
            dnn_input.append(self.dense_features)
            
        #MLP
        dnn_input=tf.concat(dnn_input,1)
        if hparams.norm is True:
            dnn_input=self.batch_norm_layer(dnn_input,self.use_norm,'dense_norm')        

        input_size=int(dnn_input.shape[-1])
        for idx in range(len(hparams.hidden_size)):
            dnn_input=tf.cond(self.use_norm, lambda: tf.nn.dropout(dnn_input,1-hparams.dropout), lambda: dnn_input)
            glorot = np.sqrt(2.0 / (input_size + hparams.hidden_size[idx]))
            W = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, hparams.hidden_size[idx])), dtype=np.float32)
            dnn_input=tf.tensordot(dnn_input,W,[[-1],[0]])
            dnn_input=tf.nn.relu(dnn_input)
            input_size=hparams.hidden_size[idx]

        glorot = np.sqrt(2.0 / (hparams.hidden_size[-1] + 1))
        W = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(hparams.hidden_size[-1], 1)), dtype=np.float32)     
        logit=tf.tensordot(dnn_input,W,[[-1],[0]])
        self.val=logit[:,0]

        self.score=tf.abs(self.val-self.label)
        self.loss=tf.reduce_mean(self.score)
        self.saver= tf.train.Saver()

    def _build_extreme_FM(self, hparams, nn_input, res=False, direct=False, bias=False, reduce_D=False, f_dim=2):
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.feature_nums
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.k])
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], hparams.k * [1], 2)
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.cross_layer_sizes):
                split_tensor = tf.split(hidden_nn_layers[-1], hparams.k * [1], 2)
                dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
                dot_result_o = tf.reshape(dot_result_m, shape=[hparams.k, -1, field_nums[0]*field_nums[-1]])
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

                if reduce_D:
                    filters0 = tf.get_variable("f0_" + str(idx),
                                               shape=[1, layer_size, field_nums[0], f_dim],
                                               dtype=tf.float32)
                    filters_ = tf.get_variable("f__" + str(idx),
                                               shape=[1, layer_size, f_dim, field_nums[-1]],
                                               dtype=tf.float32)
                    filters_m = tf.matmul(filters0, filters_)
                    filters_o = tf.reshape(filters_m, shape=[1, layer_size, field_nums[0] * field_nums[-1]])
                    filters = tf.transpose(filters_o, perm=[0, 2, 1])
                else:
                    filters = tf.get_variable(name="f_"+str(idx),
                                         shape=[1, field_nums[-1]*field_nums[0], layer_size],
                                         dtype=tf.float32)
                # dot_result = tf.transpose(dot_result, perm=[0, 2, 1])
                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
                
                # BIAS ADD
                if bias:
                    b = tf.get_variable(name="f_b" + str(idx),
                                    shape=[layer_size],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                    curr_out = tf.nn.bias_add(curr_out, b)

                curr_out = self._activate(curr_out, hparams.cross_activation)
                
                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
                
                if direct:

                    direct_connect = curr_out
                    next_hidden = curr_out
                    final_len += layer_size
                    field_nums.append(int(layer_size))

                else:
                    if idx != len(hparams.cross_layer_sizes) - 1:
                        next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                        final_len += int(layer_size / 2)
                    else:
                        direct_connect = curr_out
                        next_hidden = 0
                        final_len += layer_size
                    field_nums.append(int(layer_size / 2))

                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)


            result = tf.concat(final_result, axis=1)
            
            result = tf.reduce_sum(result, -1)

            return result