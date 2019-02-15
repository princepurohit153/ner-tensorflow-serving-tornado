# encoding=utf8

import os
import codecs
import yaml
import pickle
import tensorflow as tf
from load_data import load_vocs, init_data
from model import SequenceLabelingModel


def export_serving_model():
    """输出tensorserving model"""
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
    
    session =model.sess
    saver = tf.train.Saver()
    
    saver.restore(session, config['model_params']['path_model'])
    
    # 输出tensorserving model 过程
    model_version = 1
    work_dir = './Model/ner_model'
    
    export_path_base =work_dir
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(model_version)))

    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    
    # 定义输入变量
    tensor_info_input_x_f1 = tf.saved_model.utils.build_tensor_info(model.input_feature_ph_dict['f1'])
    tensor_info_weight_dropout_ph_dict_f1 = tf.saved_model.utils.build_tensor_info(model.weight_dropout_ph_dict['f1'])
    tensor_info_dropout_rate_ph = tf.saved_model.utils.build_tensor_info(model.dropout_rate_ph)
    tensor_info_rnn_dropout_rate_ph = tf.saved_model.utils.build_tensor_info(model.rnn_dropout_rate_ph)

    tensor_info_logits = tf.saved_model.utils.build_tensor_info(model.logits)
    tensor_info_actual_length = tf.saved_model.utils.build_tensor_info(model.sequence_actual_length)
    tensor_info_transition_params = tf.saved_model.utils.build_tensor_info(model.transition_params)
    
    
    # 构建过程
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input_x_f1': tensor_info_input_x_f1,
                    'weight_dropout_ph_dict_f1': tensor_info_weight_dropout_ph_dict_f1,
                    'dropout_rate_ph':tensor_info_dropout_rate_ph,
                    'rnn_dropout_rate_ph':tensor_info_rnn_dropout_rate_ph
                    },
                    
            outputs={'transition_params': tensor_info_transition_params,
                     'logits': tensor_info_logits,
                     'sequence_actual_length': tensor_info_actual_length,
                     },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    
    builder.add_meta_graph_and_variables(
        session, [tf.saved_model.tag_constants.SERVING],
        signature_def_map = {
            'ner_predict': prediction_signature
        },
        main_op = tf.tables_initializer(),
        strip_default_attrs = True
    )
    
    builder.save()

    print('Done exporting!')
    
    
    
export_serving_model()  

