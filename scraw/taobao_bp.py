# coding=utf-8
'''
@Time     : 2023/10/19 20:42:12
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib

import json
import os
import time
from hashlib import md5

import requests
import re
from pathlib import Path

# # 更换 print（） 的默认编码
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#
# # 禁用 https 的安全证书警告
# from requests.packages import urllib3
# urllib3.disable_warnings()

SEPARATOR = '@#$%'
requests.DEFAULT_RETRIES = 5


class GeneratePbUrl:
    def generatePB(self, items):
        """
        根据 字典列表[{'sku_id': skuId, 'num': num, 'item_id':item_id, ...},]生成对应Pb链接
        :param items:
        :return:
        """
        if not isinstance(items, list):
            raise ValueError("generatePB的参数 不是 list 类型")
        # urls = []
        buyParams = []
        for key, value in enumerate(items):
            if value['sku_id'] == '0':
                buyParam = f"{value['item_id']}_{value['num']}"
                # bpurl = f"https://h5.m.taobao.com/cart/order.html?buyParam={value['item_id']}_{value['num']}"
            else:
                buyParam = f"{value['item_id']}_{value['num']}_{value['sku_id']}"
                # bpurl = f"https://h5.m.taobao.com/cart/order.html?buyParam={value['item_id']}_{value['num']}_{value['sku_id']} "
            # urls.append(f"{value['title']}~~" + bpurl)
            buyParams.append(buyParam)
        current_group_pb = f"https://h5.m.taobao.com/cart/order.html?buyParam={','.join(buyParams)}"
        return current_group_pb

    def getSkusfromShortmsg(self, cookie, shortstring=None):
        """
        目前只支持 非优惠卷类 商品口令提取，
        一个pb能生成一个订单，
        一个pb可以支持多个商品，
        :param shortstring: 淘口令即淘宝分享的地址
        :return:
        """
        itemId_temp = re.findall(r'\?id=(.*?)&', shortstring)
        if len(itemId_temp) == 0:
            # 淘口令 获取 真实链接
            try:
                share_url = re.findall(r'https://[^\s]+', shortstring)[0]
            except Exception as e:
                raise ValueError("口令不正确！(未匹配到https)")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            }
            html = requests.get(share_url, headers=headers,
                                allow_redirects=False)
            real_url = re.findall("var url = '(.*?)'", html.text)
            if len(real_url) == 0:
                raise ValueError("口令不正确")
            # 真实链接获取重定向后的链接
            real_url = real_url[0]
            real_url_tail = real_url.split('.htm?')
            if len(real_url_tail) != 2:
                raise ValueError("口令失效")
            real_url_tail = real_url_tail[1]
            itemId_temp = re.findall(r'id=(.*?)&', real_url_tail)
            if len(itemId_temp) == 0:
                raise ValueError(f"口令不正确（{real_url_tail}）")
        itemId = itemId_temp[0]
        taobaoItemsSkuObj = TaobaoItemsSku()
        item_title, props, skuList, quantityList = taobaoItemsSkuObj.getSkuList(
            itemId, cookie)
        return taobaoItemsSkuObj, item_title, props, skuList, quantityList


class TaobaoItemsSku:
    def __init__(self):
        self.choosed_prop = []
        self.skuId = ''
        self.detail_url = ''
        self.itemId = None
        self.cookies = ''
        self.skuList = None
        self.props = None
        self.item_title = None
        self.quantityList = None
        self.try_num = 2

    def setCookie(self, cookie=''):
        self.cookies = cookie

    def getSkuList(self, itemId, cookie, save=True):
        self.cookies = cookie
        self.detail_url = f'https://detail.tmall.com/item.htm?id={itemId}'
        if itemId and itemId.isdigit():
            self.itemId = itemId
        else:
            raise ValueError(
                f"itemId:{itemId} has some words which is not digit !")
        # 解密sign
        try:
            # d8e5ec5ce90a928e9ec12c5f4dec85b4
            h5_token = re.findall('_m_h5_tk=(.*?)_', cookie)[0]
        except:
            h5_token = ""
        g = '12574478'
        i = round(time.time() * 1000)
        data = '{\"id\":\"%s\",\"detail_v\":\"3.3.2\",\"exParams\":\"{\\\"id\\\":\\\"%s\\\",\\\"queryParams\\\":\\\"id=%s\\\",\\\"domain\\\":\\\"https://detail.tmall.com\\\",\\\"path_name\\\":\\\"/item.htm\\\"}\"}' % (
            self.itemId, self.itemId, self.itemId)

        # # js逆向解析构造 sign，
        # # 搜索“_m_h5_tk怎么获取”，无意发现帖子   https://www.freesion.com/article/58941316871/
        # # 介绍说这个js函数就是生成字符串的md5值，也就是sign是拼接字符串的md5的值
        # # py方式更简单
        # with open('js/sign_tb.js', 'r', encoding='utf-8') as f:
        #     jscode = f.read()
        # signValue = execjs.compile(jscode).call('h', signKey)

        sign = self.__getSign(data=data, h5_token=h5_token, time=i, appKey=g)
        params = {'jsv': '2.6.1', 'appKey': g, 't': i, 'sign': sign, 'api': 'mtop.taobao.pcdetail.data.get',
                  'v': '1.0', 'isSec': 0, 'ecode': 0, 'timeout': 10000, 'dataType': 'json', 'valueType': 'string',
                  'ttid': '2022@taobao_litepc_9.17.0', 'AntiFlood': 'true', 'AntiCreep': 'true',
                  'preventFallback': 'true', 'type': 'json', 'data': data}

        # params = f"jsv=2.6.1&appKey={g}&t={i}&sign={sign}&api=mtop.taobao.pcdetail.data.get&v=1.0&isSec=0&ecode=0" \
        #          f"&timeout=10000&ttid=2022@taobao_litepc_9.17.0&AntiFlood=true&AntiCreep=true&preventFallback=true" \
        #          f"&type=json&data={data}"
        url = f"https://h5api.m.tmall.com/h5/mtop.taobao.pcdetail.data.get/1.0/?"
        headers = {
            'content-type': 'application/x-www-form-urlencoded',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.42',
            'Accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN,zh;q=0.9,en-GB;q=0.8,en;q=0.7',
            'cache-control': 'no-cache',
            'origin': 'https://detail.tmall.com/',
            'referer': 'https://detail.tmall.com/',
            'cookie': cookie
        }
        html = requests.get(url, headers=headers, params=params)
        res_json = json.loads(html.text)
        flag = res_json['ret'][0] if len(
            res_json['ret']) == 1 else res_json['ret'][1]
        if "令牌过期" in flag:
            cookiejar = html.cookies
            _m_h5_tk = cookiejar['_m_h5_tk']
            _m_h5_tk_enc = cookiejar['_m_h5_tk_enc']
            self.cookies = re.sub(r'_m_h5_tk=(.*?);',
                                  f'_m_h5_tk={_m_h5_tk};', self.cookies)
            self.cookies = re.sub(r'_m_h5_tk_enc=(.*?);',
                                  f'_m_h5_tk_enc={_m_h5_tk_enc};', self.cookies)
            # with open('../temps/cookies/cookie_bp_TB.txt', 'w', encoding='utf8') as bp_cookie:
            #     bp_cookie.write(self.cookies)
            return self.getSkuList(itemId, self.cookies)
        elif "令牌为空" in flag:
            cookiejar = html.cookies
            _m_h5_tk = cookiejar['_m_h5_tk']
            _m_h5_tk_enc = cookiejar['_m_h5_tk_enc']
            self.cookies = re.sub(r'_m_h5_tk=(.*?);', f'', self.cookies)
            self.cookies = re.sub(r'_m_h5_tk_enc=(.*?);', f'', self.cookies)
            self.cookies += f';_m_h5_tk={_m_h5_tk};_m_h5_tk_enc={_m_h5_tk_enc};'
            # with open('./temps/cookies/cookie_bp_TB.txt', 'w', encoding='utf8') as bp_cookie:
            #     bp_cookie.write(self.cookies)
            return self.getSkuList(itemId, self.cookies)

        elif "调用成功" in flag:
            if save:
                save_path = Path(__file__).parent / "get_data"
                save_path.mkdir(parents=True, exist_ok=True)
                with open(save_path / f"{itemId}-{i}.json", 'w+', encoding='utf8') as file:
                    json.dump(res_json, file, ensure_ascii=False, indent=2)
            item_title, props, skuList, quantityList = self.__getPropsfromResponse(
                res_json)
            return item_title, props, skuList, quantityList

        # elif "哎哟喂,被挤爆啦,请稍后重试" in flag:
        #     x5secdata, x5sec = self._getx5sec(res_json)
        #     self.cookies = self.cookies + 'x5secdata=' + x5secdata
        #     return self.getSkuList(itemId, self.cookies)

        else:
            cookiejar = html.cookies
            print(cookiejar)
            # _m_h5_tk = cookiejar['_m_h5_tk']
            # _m_h5_tk_enc = cookiejar['_m_h5_tk_enc']
            # print(_m_h5_tk)
            raise ValueError(flag)

    def __getPropsfromResponse(self, resp_data):
        data = resp_data['data']
        self.item_title = data['item']['title']
        self.props = []
        # 有些商品没有属性，比如优惠卷，返回的skuBase的数据结构为：{'components': [], 'skus': []}，没有props
        if not data['skuBase'].get('props'):
            data['skuBase'].update(
                {'props': [{'hasImage': 'false', 'name': '', 'pid': '0',
                            'values': [{'name': "", 'sortOrder': "1", 'vid': "0"}]}]})
            data['skuBase']['skus'].append({'propPath': "0:0", 'skuId': "0"})
        for prop in data['skuBase']['props']:
            t = {'name': prop['name'], 'id': prop['pid'],
                 'skuVid': {}, 'skuTexts': []}
            for i in prop['values']:
                t['skuVid'][i['name']] = i['vid']
                t['skuTexts'].append(i['name'])
            self.props.append(t)

        self.skuList = {}
        for sku in data['skuBase']['skus']:
            self.skuList[sku['propPath']] = sku['skuId']

        self.quantityList = {}
        for key, value in self.skuList.items():
            self.quantityList[value] = data['skuCore']['sku2info'][value]['quantity']
        return self.item_title, self.props, self.skuList, self.quantityList

    def getSkuIdfromProps(self, choose):
        self.choosed_prop = choose
        if len(choose) != len(self.props) or not self.props:
            print(f"too many args, only need {len(choose)} args !")
            raise ValueError(f"【错误警告】 未获取其商品属性!")
        skuKey = ''
        for i in range(len(self.props)):
            a = choose[i]
            a_id = self.props[i]['id']
            a_sku = self.props[i]['skuVid'][a]
            skuKey += a_id + ":" + a_sku + ";"
        skuKey = skuKey[:-1]
        if not self.skuList.get(skuKey):
            raise ValueError("【错误警告】 该商品 无 此属性组合！！")
        self.skuId = self.skuList[skuKey]
        return self.skuId

    def __getSign(self, data, h5_token, time, appKey='12574478'):
        md5Obj = md5()
        md5Obj.update(
            f"{h5_token}&{time}&{appKey}&{data}".encode(encoding="utf8"))
        sign = str(md5Obj.hexdigest())
        return sign


if __name__ == '__main__':
    short = '淘宝】https://m.tb.cn/h.5O51g0t?tk=BSN7Wf8kqoX CZ0001'
    # with open('../temps/cookies/cookie_bp_TB.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     lines = (line.strip() for line in lines)
    #     cookie = "".join(lines)
    # print(cookie)
    cookie = "lid=smilexd2510; cna=9/QRHdFjXiYCASSf7/EhgNzw; dnk=smilexd2510; uc1=cookie16=UtASsssmPlP%2Ff1IHDsDaPRu%2BPw%3D%3D%26pas=0%26cookie14=UoYelzxqtBD9Fg%3D%3D%26cookie15=U%2BGCWk%2F75gdr5Q%3D%3D%26cookie21=U%2BGCWk%2F7p4mBoUyS4E9C%26existShop=false; uc3=nk2=EFOacxalClWOvx0%3D%26vt3=F8dD3CQ7yPfdG1kNV3o%3D%26lg2=Vq8l%2BKCLz3%2F65A%3D%3D%26id2=UUGgq6rW%2BCw8ow%3D%3D; tracknick=smilexd2510; _l_g_=Ug%3D%3D; uc4=nk4=0%40EoqZYFKyz%2BtJxPel90XlwNL7d%2F0YjQ%3D%3D%26id4=0%40U2OXkZ%2B0LWcqN9TTN%2BNa26XJW4OA; unb=2926012310; lgc=smilexd2510; cookie1=BxeAZDoqEb0po2BBw6XrjYR3tjXhBztoPfNyGRbpbGQ%3D; login=true; cookie17=UUGgq6rW%2BCw8ow%3D%3D; cookie2=1c5506f1686b8647baa73c209a963e2d; _nk_=smilexd2510; sgcookie=E100bXdaZkgcNNrIZjIDO43fugQMrSwRulVGYVJtoTZ11rOSORNgdqnyKTNoHLFTvCkvN0B5fcWI6shARi91bdIEWTcaQRUsr15qG7r%2BjsNaki0%3D; cancelledSubSites=empty; sg=005; t=514ba0254712de724d8b85a2c593940d; csg=188d35f2; _tb_token_=3865ee5333e8e; _m_h5_tk=1d74c9af0fe1f4a351191096eac2dbda_1702047561560; _m_h5_tk_enc=fd792a6a832f2f565d662fa92776d2c0; isg=BAAA-at-cbAi7QxLAl5ShBJN0Y7SieRTw3g7sXqRJZuf9aAfI5nE49YEDd013pwr; l=fBTrDmocNmaxOxFpBO5Inurza77TCIRb4kPzaNbMiIEGa6LF6eejyNCTou0BJdtfgTCcyeKrWDM-1dLHR3m-2xDDB_5LaCom3xv9QaVb5; tfstk=ePmBfhvoh6fImS_h1B9ZG24lAXqSRXt2RTw-nYIFekELFgMmNvrre7o7VfcTyDrELftSnjaUy0PrV9msGgSeteDRVurJgI-20vX3qudVWMmb5vDEFI190nknpSoE60t2K7eeEieCZsRFWAO1XGsSeej8y1aLjcwQWPkQ6WNpUJaCfvN12RmLd0Fxp5sR2glN0Si4Wa6_ogw_gd91ra0_TRS-9fpi5yegQmJ6C6_uJR2_Ud91rN4LIRfWCd_mf"
    taobaoItemsSkuObj, item_title, props, skuList, quantityList = GeneratePbUrl(
    ).getSkusfromShortmsg(cookie, short)
    chooseProp = []
    for key, i in enumerate(props):
        print(f"属性{key + 1}--{i['name']}", ":", *tuple(i['skuTexts']))
    #     chooseProp.append(input('请选择' + str(i['name']) + ':'))
    # skuId = taobaoItemsSkuObj.getSkuIdfromProps(chooseProp)
    # prop = {'sku_id': skuId, 'num': 1, 'item_id': taobaoItemsSkuObj.itemId}
    # bpUrl = GeneratePbUrl().generatePB([prop])
    # print("bp:", bpUrl)

    # # 获取随机ua
    # from faker import Faker
    # uas = Faker()
    # print(uas.user_agent())
