# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-7-23 02:10
# Tool ：PyCharm
import json
import time

import requests

ques = '不是你刚刚说的这个故事吧'
data = {
    "model": "string",
    "messages": [
        {
            "role": "user",
            "content": ques.strip(" ")
        }
    ],
    "temperature": 0,
    "top_p": 0,
    "max_length": 0,
    "stream": False
}
session = requests.session()
headers = {
    "content-type": "application/json"
}

q_t = time.strftime("%Y-%m-%d %X")
print(f"【{q_t}】问： {ques}")

question = session.post(f'http://111.45.28.184:7860/chat', data=json.dumps(data)).text
response = json.loads(question)

for message in response['choices']:
    t = time.gmtime(response['created'])
    t = time.strftime("%Y-%m-%d %X", t)
    print(f"【{t}】答： {message['message']['content']}")
