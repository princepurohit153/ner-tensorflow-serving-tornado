NER tf model with tfserving+tornado 
===================================

This project using the trained ner model to deploy a web service using tfserving
and tornado for high concurrent request.


### Deploy procedure 
    
Deploy a ner tfserving service contains the following three main procedures

step 1: Train and export ner model
```
# prepare trainchange config.yml and training model

python train.py

```

step 2: Load exported model with standard TensorFlow ModelServer
```
# export tfServing model format

python export_model.py

# Serve the Model

bash ./excute_tfServing_service.sh

```

step 3: Using client to test the server
```
# direct load the tfserver model to check the correctness of the exported model

python tfServing_model_predict.py

# call tfService using the client

python load_ner_tfserving_model.py

```

step 4: Using Tornado to make the ModelServer available for web serving
```
python tfServing_tonado.py

```

step 5: Test web ModelServer with high concurrent request
```
python api_test.py

```


### Requirements
    
The following are the main requirements, to run the program it need to install
 other packages with the exceptions
    
- Python >= 2.7
- docker
- [tf-serving](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md)
- tornado


### Reference

- [https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/serving_basic.md](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/serving_basic.md)

