#encoding:utf-8

import os, re
import requests,json,sys,time

def get_web_data(url, content):
    r = requests.post(url,content)
    if r.status_code!=200:
        raise ValueError(u'请求失败,错误代码%s'.format(r.status_code))
    return r.text


def timer(fun):
    def deco(*args, **kwargs):
        stime = time.time()
        out = fun(*args, **kwargs)
        time_consume = time.time()-stime
        print(time_consume)
        return out
    return deco

    
def enable_parallel(fun,datalist,processnum=None):
    """
    :param fun: function 
    :param datalist: list, [str,str,...]
    :param processnum: cpu cores 
    :return: list, [str,str,...], the str data need to convert to list or dict using eval or json.loads
    """
    import multiprocessing
    from multiprocessing import cpu_count
    multiprocessing.freeze_support()
    
    if os.name == 'nt':
        raise NotImplementedError("parallel mode only supports posix system")
    else:
        from multiprocessing import Pool
    if processnum is None:
        processnum = cpu_count()
    pool = Pool(processnum)
    paradata = pool.map(fun, datalist)
    disable_parallel(pool)
    return paradata

def disable_parallel(pool):
    if pool:
        pool.close()
    
#@timer
def ner_predict(inputdata):
    """Test the tornado model service"""
    url = 'http://192.168.XX.XX:6800/wordner'# 改成tornado服务部署地址
    content = {'input':inputdata}
    outdata = get_web_data(url, content)
    return outdata

def main(sentences, parallel=True):
    import time
    stime = time.time()

    if parallel:
        paradata = enable_parallel(ner_predict, sentences)
        out = [json.loads(data) for data in paradata]
        print out, type(paradata)
    else:
        data = ner_predict(''.join(sentences))
        print data

    print time.time() - stime


if __name__=="__main__":
    indata = sys.argv[1:]
    sentences = [u'''因"反复抽搐1月余,再发伴皮疹3天"于2015-05-18 11:24收入本科。''']
    if len(indata)==2:
        sentences, parallel = indata
    else:
        print('''command should be: python api_test.py u"因反复抽搐1月余,再发伴皮疹3天入院" True/False ''')

    if indata:
        sentences = [indata]

    main(sentences)
