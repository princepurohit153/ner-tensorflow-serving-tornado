#!/bin/bash
# 启动 NER tf ModelServer 服务脚本
# Useage：sudo bash excute.sh
echo '*****kill existed docker service !!*****'
sudo ps -ef|grep docker|grep -v grep|awk '{print "kill -9 "$2}'|sh
relative_path='/Model/ner_model'
path=$PWD$relative_path
echo -e "*****start tensorsering: ner model!*****\ncurrent ner model path is : ${path}\n" >> log.txt
service docker start
docker run -p 8500:8500 --mount type=bind,source=$path,target=/models/ner -e MODEL_NAME=ner -t tensorflow/serving >> log.txt &
echo -e '*****succeed start tensorsering: ner model!*****\n'