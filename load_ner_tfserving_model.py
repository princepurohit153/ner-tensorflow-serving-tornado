# encoding=utf8

import re
import json
import codecs
import grpc
import tensorflow as tf
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import yaml
import pickle
import math
from load_data import load_vocs, init_data

def check_char(char):
    if not char.strip():
        char = 'Ю'
    elif char=='\n':
        char = 'Ж'
    return char
   
def recheck_char(char):
    if char == 'Ю':
        char = ' '
    elif char=='Ж':
        char = '\n'
    return char

def load_parameters():
    """load parameters from config file"""
    
    with open('./config.yml') as file_config:
        config = yaml.load(file_config)

    feature_names = ['f1']
    use_char_feature = config['model_params']['use_char_feature']

    path_vocs = []
    
    for feature_name in feature_names:
        path_vocs.append(config['data_params']['voc_params'][feature_name]['path'])
    path_vocs.append(config['data_params']['voc_params']['label']['path'])
    vocs = load_vocs(path_vocs)
    
    sep_str = config['data_params']['sep']
    assert sep_str in ['table', 'space']
    sep = '\t' if sep_str == 'table' else ' '
    max_len = config['model_params']['sequence_length']
    word_len = config['model_params']['word_length']
    
    return feature_names, sep, vocs, max_len, use_char_feature, word_len

def load_ner_service(feed_dict):
    """Using tfServing Client do the prediction
    paramters：
        feed_dict: dict, contains the feed data
    """
    # load ner tfserving
    # @Warning: necessary to set grpc.insecure_channel options to increase lightning gRPC block size from default 4MB to 50MB
    host = 'localhost:8500'
    channel = grpc.insecure_channel(host, options=[('grpc.max_receive_message_length',1024*1024*50)])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'ner'
    request.model_spec.signature_name='ner_predict'
    
    # predict
    for feature in feed_dict.keys():
        request.inputs[feature].CopyFrom(
            tf.contrib.util.make_tensor_proto(feed_dict[feature],
                                              shape=feed_dict[feature].shape))
    result_future = stub.Predict.future(request,20.0)
    
    # reshape the tfserving data and using viterbi to decode the result
    transition_shape = map(lambda x:int(x.size), result_future.result().outputs["transition_params"].tensor_shape.dim)
    transition_params = np.reshape(np.array(result_future.result().outputs["transition_params"].float_val), transition_shape)
    
    logits_shape = map(lambda x:int(x.size), result_future.result().outputs["logits"].tensor_shape.dim)
    logits = np.reshape(np.array(result_future.result().outputs["logits"].float_val), logits_shape)
    
    # be careful the sequence_actual_length is int type, using the int_val to get the value
    sequence_actual_length = result_future.result().outputs["sequence_actual_length"].int_val
    
    return logits, sequence_actual_length, transition_params

    
def pre_sentence(sentences):
    
    """Change sentences to the needed sentences
    paramters：
        sentences: string,
        senlist: list,[str,str,...]
    """
    
    charlist, senlist = [],[]
    sentences = data_clean(sentences)
    for char in sentences:
        #char = check_char(char)
        charlist.append([char])
        #print 'char',char
        if char in [u'。', u';']:
            senlist.append(charlist)
            charlist=[]
    if charlist:
        senlist.append(charlist)
    
    return senlist

    
def data_clean(strs):
    """
    :param strs:string documents 
    :return: string documents
    """
    pattern = [(u'，',','),(u'？','?'),(u'：',':'),(u'“','"'),(u'”','"'),(u"＞",">"),
               (u"‘","'"),(u"’","'"),(u"（","("),(u"）",")"),(u"《","<"),(u"＜","<"),
               (u"》",">"),(u"！","!"),(u"；",";"),(u'【',u'['),(u'】',u']'),(u'％',u'%'),
               (u'﹪',u'%'),(u'➕',u'+'),(u'➖',u'-'),(u'＋',u'+'),(u'．',u'.')]
    
    for x,y in pattern:
        strs = re.sub(x,y,strs)
    return strs


def predict(testlist):
    """Prepare the model input data type
    paramters：
        testlist: list,[[[u'T'], [u':'],...]
    return result_sequences: list,
    """
    feature_names, sep, vocs, max_len, use_char_feature, word_len = load_parameters()
    
    data_dict = init_data( feature_names=feature_names,
                           sep=sep,test_sens=testlist,vocs=vocs, max_len=max_len,
                           model='test',use_char_feature=use_char_feature,word_len=word_len)
    
    # 生成模型feed data 
    data_count = data_dict['f1'].shape[0]
    nb_test = int(math.ceil(data_count /16.0))
    result_sequences = []  # 标记结果
    for i in range(nb_test):
        feed_dict = dict()
        batch_indices = np.arange(i * 16, (i + 1) * 16) \
            if (i+1)*16 <= data_count else \
            np.arange(i*16, data_count)
        batch_data = data_dict['f1'][batch_indices]
        item = {'input_x_f1': batch_data}
        feed_dict.update(item)
        # dropout
        item = {'weight_dropout_ph_dict_f1': np.array(0.0, dtype=np.float32)}
        
        feed_dict.update(item)
        
        feed_dict.update({'dropout_rate_ph':np.array(0.,dtype=np.float32), 'rnn_dropout_rate_ph': np.array(0.,dtype=np.float32)})
        
        # viterbi decode procedure
        logits, sequence_actual_length, transition_params = load_ner_service(feed_dict)
        
        for logit, seq_len in zip(logits, sequence_actual_length):
            logit_actual = logit[:seq_len]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(logit_actual, transition_params)
            result_sequences.append(viterbi_sequence)

    return result_sequences
    

def main(inputdata):
    vocs = load_parameters()[2]
    testlist = pre_sentence(inputdata)
    result_sequences = predict(testlist)
    #print 'result_sequences',type(result_sequences)
    
    label_voc = dict()
    for key in vocs[-1]:
        label_voc[vocs[-1][key]] = key

    outlist = []
    for i, sentence in enumerate(testlist):
        templist = []
        for j, item in enumerate(sentence):
            char = item[0]
            if j < len(result_sequences[i]):
                out = [char, label_voc[result_sequences[i][j]]]
            else:
                out = [char, 'O']
            templist.append(out)
        outlist.append(templist)
    #print 'outlist', str(outlist).encode().decode('unicode_escape')
    return outlist

    
if __name__=="__main__":
    inputdata = u"""2014-07-0611:41首次病程记录病例特点：患者:XXX，男，50岁，主诉以“右下肢肿胀2天”入院。"""
    main(inputdata)

        

        

    
    
    