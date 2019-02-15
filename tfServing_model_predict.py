# encoding=utf8

import re
import json
import codecs
import grpc
import numpy as np
import tensorflow as tf

import yaml
import pickle
import math
from load_data import load_vocs, init_data
        
def pre_sentence(sentences):
    """将sentence转换为需要的模型序列"""
    charlist, senlist = [],[]
    sentences = data_clean(sentences)
    for char in sentences:
        char = check_char(char)
        charlist.append([char])
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
        #strs = strs.replace(x,y)
        strs = re.sub(x,y,strs)
    return strs



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

    # 加载vocs
    path_vocs = []
    
    for feature_name in feature_names:
        path_vocs.append(config['data_params']['voc_params'][feature_name]['path'])
    path_vocs.append(config['data_params']['voc_params']['label']['path'])
    vocs = load_vocs(path_vocs)

    # 加载数据
    sep_str = config['data_params']['sep']
    assert sep_str in ['table', 'space']
    sep = '\t' if sep_str == 'table' else ' '
    max_len = config['model_params']['sequence_length']
    word_len = config['model_params']['word_length']
    
    return feature_names, sep, vocs, max_len, use_char_feature, word_len
    
    
def pre_feed_data(testlist):
    # 加载配置文件
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
        
        feed_dict.update({'dropout_rate_ph':np.array(0.0,dtype=np.float32), 'rnn_dropout_rate_ph': np.array(0.0,dtype=np.float32)})
        
        print 'feed_dict_tfserving', feed_dict
        yield feed_dict
    
def tf_serving_predict(feed_dict):
    """直接加载tf model模型预测"""
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    meta_graph_def = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], './Model/ner_model/1')
    signature = meta_graph_def.signature_def

    #
    tensor_info_input_x_f1 = signature['ner_predict'].inputs['input_x_f1'].name
    tensor_info_weight_dropout_ph_dict_f1 = signature['ner_predict'].inputs['weight_dropout_ph_dict_f1'].name
    tensor_info_dropout_rate_ph = signature['ner_predict'].inputs['dropout_rate_ph'].name
    tensor_info_rnn_dropout_rate_ph = signature['ner_predict'].inputs['rnn_dropout_rate_ph'].name
    
    tensor_info_logits=signature['ner_predict'].outputs['logits'].name
    tensor_info_transition_params=signature['ner_predict'].outputs['transition_params'].name
    tensor_info_actual_length=signature['ner_predict'].outputs['sequence_actual_length'].name
    
    #
    input_x_f1 = session.graph.get_tensor_by_name(tensor_info_input_x_f1)
    weight_dropout_ph_dict_f1 = session.graph.get_tensor_by_name(tensor_info_weight_dropout_ph_dict_f1)
    dropout_rate_ph = session.graph.get_tensor_by_name(tensor_info_dropout_rate_ph)
    rnn_dropout_rate_ph = session.graph.get_tensor_by_name(tensor_info_rnn_dropout_rate_ph)
    logits = session.graph.get_tensor_by_name(tensor_info_logits)
    transition_params = session.graph.get_tensor_by_name(tensor_info_transition_params)
    sequence_actual_length = session.graph.get_tensor_by_name(tensor_info_actual_length)
    # print 'feed_dict_tfserving1',feed_dict
    logits, transition_params, sequence_actual_length=session.run([logits, transition_params, sequence_actual_length], feed_dict={input_x_f1:feed_dict['input_x_f1'],
                                          weight_dropout_ph_dict_f1:feed_dict['weight_dropout_ph_dict_f1'],
                                          dropout_rate_ph:feed_dict['dropout_rate_ph'],
                                          rnn_dropout_rate_ph:feed_dict['rnn_dropout_rate_ph']})
    
    # print 'sequence_actual_length', sequence_actual_length
    result_sequences = []
    for logit, seq_len in zip(logits, sequence_actual_length):
        logit_actual = logit[:seq_len]
        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
            logit_actual, transition_params)
        result_sequences.append(viterbi_sequence)
    # print 'result_sequences',result_sequences    
    return result_sequences


def main(inputdata):
    feature_names, sep, vocs, max_len, use_char_feature, word_len = load_parameters()
    
    testlist = pre_sentence(inputdata)
    feed_dict_iters = pre_feed_data(testlist)    
    for feed_dict in feed_dict_iters:
        result_sequences = tf_serving_predict(feed_dict)
        
        label_voc = dict()
        for key in vocs[-1]:
            label_voc[vocs[-1][key]] = key
        
        outlist = []
        for i, sentence in enumerate(testlist):
            for j, item in enumerate(sentence):
                outlist.append([item+[label_voc[result_sequences[i][j]]]])
        
        print 'outlist', outlist
 
        

if __name__=="__main__":
    inputdata = u'T:38.0，P:96次/分，'
    main(inputdata)



