from src import misc_utils as utils
from models import CIN
import tensorflow as tf
from imp import reload
def build_model(hparams):
    tf.reset_default_graph()
    if hparams.model=='CIN':
        model=CIN.Model(hparams)        
    else:
        raise "Model error"
    config_proto = tf.ConfigProto(log_device_placement=0,allow_soft_placement=0)
    config_proto.gpu_options.allow_growth = True
    sess=tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    model.set_Session(sess)
    
    return model