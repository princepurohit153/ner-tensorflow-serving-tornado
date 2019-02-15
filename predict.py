#!/usr/bin/env python
# -*- encoding: utf-8 -*-
__author__ = 'Hwa.chun'
"""
    标记文件
"""

import re
import codecs
import yaml
import pickle
import tensorflow as tf
from load_data import load_vocs, init_data
from model import SequenceLabelingModel

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


def predict(testlist):
    # 加载配置文件
    with open('./config.yml') as file_config:
        config = yaml.load(file_config)

    feature_names = config['model_params']['feature_names']
    use_char_feature = config['model_params']['use_char_feature']

    # 初始化embedding shape, dropouts, 预训练的embedding也在这里初始化)
    feature_weight_shape_dict, feature_weight_dropout_dict, \
    feature_init_weight_dict = dict(), dict(), dict()
    for feature_name in feature_names:
        feature_weight_shape_dict[feature_name] = \
            config['model_params']['embed_params'][feature_name]['shape']
        feature_weight_dropout_dict[feature_name] = \
            config['model_params']['embed_params'][feature_name]['dropout_rate']
        path_pre_train = config['model_params']['embed_params'][feature_name]['path']
        if path_pre_train:
            with open(path_pre_train, 'rb') as file_r:
                feature_init_weight_dict[feature_name] = pickle.load(file_r)
    # char embedding shape
    if use_char_feature:
        feature_weight_shape_dict['char'] = \
            config['model_params']['embed_params']['char']['shape']
        conv_filter_len_list = config['model_params']['conv_filter_len_list']
        conv_filter_size_list = config['model_params']['conv_filter_size_list']
    else:
        conv_filter_len_list = None
        conv_filter_size_list = None

    # 加载vocs
    path_vocs = []
    if use_char_feature:
        path_vocs.append(config['data_params']['voc_params']['char']['path'])
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
    data_dict = init_data(
        path=config['data_params']['path_test'], feature_names=feature_names, sep=sep,test_sens=testlist,
        vocs=vocs, max_len=max_len, model='test', use_char_feature=use_char_feature,
        word_len=word_len)

    # 加载模型
    model = SequenceLabelingModel(
        sequence_length=config['model_params']['sequence_length'],
        nb_classes=config['model_params']['nb_classes'],
        nb_hidden=config['model_params']['bilstm_params']['num_units'],
        num_layers=config['model_params']['bilstm_params']['num_layers'],
        feature_weight_shape_dict=feature_weight_shape_dict,
        feature_init_weight_dict=feature_init_weight_dict,
        feature_weight_dropout_dict=feature_weight_dropout_dict,
        dropout_rate=config['model_params']['dropout_rate'],
        nb_epoch=config['model_params']['nb_epoch'], feature_names=feature_names,
        batch_size=config['model_params']['batch_size'],
        train_max_patience=config['model_params']['max_patience'],
        use_crf=config['model_params']['use_crf'],
        l2_rate=config['model_params']['l2_rate'],
        rnn_unit=config['model_params']['rnn_unit'],
        learning_rate=config['model_params']['learning_rate'],
        use_char_feature=use_char_feature,
        conv_filter_size_list=conv_filter_size_list,
        conv_filter_len_list=conv_filter_len_list,
        word_length=word_len,
        path_model=config['model_params']['path_model'])
    saver = tf.train.Saver()
    saver.restore(model.sess, config['model_params']['path_model'])

    # print('data_dict', data_dict)
    # 标记
    result_sequences = model.predict(data_dict)

    #print('result_sequences', result_sequences)

    # 输出结果
    label_voc = dict()
    for key in vocs[-1]:
        label_voc[vocs[-1][key]] = key

    outlist = []
    for i, sentence in enumerate(testlist):
        templist = []
        for j, item in enumerate(sentence):
            #char = recheck_char(item[0])
            char = item[0]
            if j < len(result_sequences[i]):
                out = [char,label_voc[result_sequences[i][j]]]
            else:
                out = [char,'O']
            templist.append(out)
        outlist.append(templist)
    return outlist


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
        strs = re.sub(x,y,strs)
    return strs

    
def main(inputdata):
    data = pre_sentence(inputdata)
    #print 'data', data
    outdata = predict(data)
    print 'outdata', str(outdata).encode().decode('unicode_escape')
    return outdata


    
if __name__ == '__main__':
    inputdata = u"""2014-07-0611:41首次病程记录病例特点：患者,主诉以“右下肢肿胀2天”入院。"""
    main(inputdata)

    
    
    