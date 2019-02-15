#encoding:utf-8
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.httpclient

import tornado.gen
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor

import json
import os
import datetime

from load_ner_tfserving_model import main


def makeResponseBody(retCode, errReason,result):
    dicRes = {}
    dicRes['retCode'] = retCode
    if retCode != 0:
        dicRes['error'] = errReason
    else:
        dicRes['data'] = result
    return json.dumps(dicRes)


class ExtractionHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(50)
    @tornado.gen.coroutine
    def post(self):
        result=[]
        try:
            #content = self.get_argument('input').encode('utf8').strip()
            content = self.get_argument('input')
        except:
            errorReason = 'Query encoding not utf-8'
            strRes = makeResponseBody(-1, errorReason,result)
            self.write(strRes)
            return
        if content == "":
            strRes = makeResponseBody(0, 'input is empty',result)
            self.write(strRes)
            return
        error, errReason,result = yield self.get_extract(content)
        strRes = makeResponseBody(error, errReason,result)
        #print strRes
        #self.render('output.html',title='Tornado GET&POST',seg=strRes)
        self.write(strRes)


    @run_on_executor
    def get_extract(self, content):
        result=[]
        try:
            #result=output(content)#分词
            result=main(content)#实体识别
            
            return 0, 'success',result
        except:
            return 1,'fail',result


if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/wordner", ExtractionHandler)],
                                    # template_path = os.path.join(os.path.dirname(__file__),"templates"),
                                    autoreload=False,
                                    debug = False
                                  )
    http_server = tornado.httpserver.HTTPServer(app)
    #http_server.bind(5900)#分词
    http_server.bind(6800)#实体识别
    http_server.start(0)
    # print(tornado.ioloop.IOLoop.initialized())
    tornado.ioloop.IOLoop.instance().start()