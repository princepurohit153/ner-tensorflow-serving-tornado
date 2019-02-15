#!/usr/bin/env python
# -*- encoding: utf-8 -*-
__author__ = 'jxliu.nlper@gmail.com'
"""
    加载数据
"""
import sys
import codecs
import pickle
import numpy as np
from utils import map_item2id


def load_vocs(paths):
    """
    加载vocs
    Args:
        paths: list of str, voc路径
    Returns:
        vocs: list of dict
    """
    vocs = []
    for path in paths:
        with open(path, 'rb') as file_r:
            vocs.append(pickle.load(file_r))
    return vocs


def load_lookup_tables(paths):
    """
    加载lookup tables
    Args:
        paths: list of str, emb路径
    Returns:
        lookup_tables: list of dict
    """
    lookup_tables = []
    for path in paths:
        with open(path, 'rb', encoding='utf-8') as file_r:
            lookup_tables.append(pickle.load(file_r))
    return lookup_tables




def init_data(feature_names, vocs, max_len, model='train',path=None, test_sens=None,
              use_char_feature=False, word_len=None, sep='\t'):
    """
    加载数据(待优化，目前是一次性加载整个数据集)
    Args:
        path: str, 数据路径
        test_sens: list, [[[u'白带常规', u 'ni', u 'S_ex_name'],[u '：', u 'w', u 'O'],[],...],[[],[],[],...],...]
        feature_names: list of str, 特征名称
        vocs: list of dict
        max_len: int, 句子最大长度
        model: str, in ('train', 'test')
        use_char_feature: bool，是否使用char特征
        word_len: None or int，单词最大长度
        sep: str, 特征之间的分割符, default is '\t'
    Returns:
        data_dict: dict
    """
    assert model in ('train','test')
    if model=='train':
        with codecs.open(path,'r',encoding='utf8') as file_r:
            sentences = file_r.read().strip().split('\n\n')
        sentences = [[j.split(sep) for j in sen.split('\n')] for sen in sentences]
    else:
        if not test_sens:
            raise ValueError('请保证测试语料非空！！！')
        sentences = test_sens

    sentences_count = len(sentences)
    print('sentences_count1',sentences_count)
    feature_count = len(feature_names)
    data_dict = dict()
    for feature_name in feature_names:
        data_dict[feature_name] = np.zeros((sentences_count, max_len), dtype='int32')
    
    #char feature
    if use_char_feature:
        data_dict['char'] = np.zeros((sentences_count,max_len,word_len),dtype='int32')
        char_voc=vocs.pop(0)
    if model == 'train':
        data_dict['label'] = np.zeros((len(sentences),max_len),dtype='int32')
    for index, items in enumerate(sentences):
        one_instance_items = []
        [one_instance_items.append([]) for _ in range(len(feature_names)+1)]
        for feature_tokens in items:
            for j in range(feature_count):
                one_instance_items[j].append(feature_tokens[j])
            if model == 'train':
                one_instance_items[-1].append(feature_tokens[-1])
        for i in range(len(feature_names)):
            data_dict[feature_names[i]][index,:]=map_item2id(
                one_instance_items[i],vocs[i],max_len)
        if use_char_feature:
            for i,word in enumerate(one_instance_items[0]):
                if i>= max_len:
                    break
                data_dict['char'][index][i,:] = map_item2id(word,char_voc,word_len)
        if model == 'train':
            data_dict['label'][index,:] = map_item2id(one_instance_items[-1],
                                                      vocs[-1],max_len)
            print('loading data: %d\r'%index)

    return data_dict
