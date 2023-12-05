# coding=utf-8
'''
@Time     : 2023/10/19 20:51:13
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib
import re
import json
import requests
import time


class slider_kill():
    def __init__(self) -> None:
        self.cookies = ""

    def _getx5sec(self, punish_data):
        x5sec_url = punish_data['data']['url']
        x5secdata = re.findall('x5secdata=(.*?)&', x5sec_url)[0]
        cookie = self.cookies + 'x5secdata=' + x5secdata
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
        resp_slide = requests.get(x5sec_url, headers=headers).text
        patren1 = ' window._config_ = (.*?);'
        result_re = re.compile(patren1, re.S).findall(resp_slide)[0]
        data_slide = json.loads(result_re)

        slideurl = "https://h5api.m.tmall.com/h5/mtop.taobao.pcdetail.data.get/1.0/_____tmd_____/slide?"
        # uuid 就是 NCTOKENSTR
        slidedata = {
            'a': data_slide['NCAPPKEY'],
            't': data_slide['NCTOKENSTR'],
            'n': "",
            'p': "{\"ncbtn\":\"62|196.60000610351562|41.60000228881836|29.600000381469727|196.60000610351562|226"
                 ".20000648498535|62|103.60000228881836\","
                 "\"umidToken\":\"G81C4D60AA872FBCE1E5566B72617655AA58BC074DA3F45963F\","
                 "\"ncSessionID\":\"895d44345ab\"}",
            'scene': "register",
            'asyn': 0,
            'lang': "cn",
            'v': 1
        }
        params = {
            'slidedata': '{"a":"X82Y__d366b44467147323c70b42d8a2d852f5","t":"9e0b43b00163ce296d4644747b6610d7","n":"225!MRrxdizWooiUgi0jJjipezFXi+lRahRPSodaVsJDjFbBTN28R1cuvdZVr6XsrQ737iydQdTX0NhOX49pxI07VDXHjVLxHBbYPpVXfozlb34djxf/oLnjDUHofeG92C3OeyKcIZECJU4KjcI4z43+/mFus4UjGjWu4MXhfidCbU4KjcI4oL3GDGBz1nQqvpeGD654foz3bUSKjMXfJ4io4WmjpB9LuKcGKyLluK1e3O6Omy7/SVTHVItRf4Ga640dDMXhfizlbUSKjxIhoLjjDlhuGOfmNIUXSP56fK+iUowtkQz3Hq4agOJM2VSOR0pPbAyMfI85dOaqqBzSnq0Ly+ZxMSQoM1bayyO7lzzg+FSq67J4nMYyeFIIQx3fujDGHw86f+dU3E9NvqkaEy0/9EJ8P44uOleNH22qfdKF+NCYY/gEHnuEVvkM8c0MOfUDandImZ55oZW7685z5qYEy+ORQxhBujSuh8zucozdb3SdjjQcdi6RFCBzfeG0QzdJCfRefw30b3ZIMp+/FN4xMclG4c9dmWeOuZ/E0ixo70Jl0wGo4zlNngLNVy0lhkuT3eJ2+f6zMpBwCju3XqLY5Xbgu6vsgFQlYwDT2/Rxfo/DZhH4YwaeCtHpnKQjnl/eK65V+BR66nBL+c3fIpgFac2Ksn2OreYOoJOHmL6Zm/3w46oY4sNeDFTp7spUKFk8t6ceEkHTOgUs0u5fa7TMQcfHubCkFRvF3foBbX2pACGnHcXsgB9tZX3rVWn/lYo7OGAH1KAUonbw3S//tgVWIgFgDkYr5r+XgRMVzo4t+h5IO5WT/yD53dSFB/s+hFJSPeFvOa7Sp38SfRrr+lZnWrENEtjT+Yf6ClbKf3LoHNycB85HlCcVxBI1JLX+EMk2UBjWAH5wohoWd6zN1Ub2HgFFdBqqUKjlHamx2GpSQkmSVPPu2TpFn7eYNsNV+gR7Zvg2KSsVgVO2prJ9f3+d9cxeGEJm3VCURo+uDIpCwxDXMUr+2bvuoOjUQkq/22YvtN64JIc+kqun5UeQ2/i0TFhi1+6P1lpgB0ewsQImCSVsl2BqRrtOFyp4xsinTElthmz3VHR0sQqxnLOXnLYuSnZHMfRfPT0mfs5WbqYo1qrBl8ULPGFJ10LQJSCmvoHxZzAqSSRVu74K5H3lEXx+QBM7Lm4sIdSmCwvmsyt8yl0LNL9mNXDsG+74Uw/YU2/jyedKUg5rmSha/jlLjrCu80zwlbHF/lw/xorB9cVZwYzGbhqMWp0ZPGDmYn5AGoXefXPPhnJwT+QBQdQTXIDk/XxCzcfiCQ3KW0YtMPdleCtkBNFdNwBodKy9WmuxWA53YXk26LNFHyZuuplTk1Q1yqAFDxt1eO7GMAZ5OAuX7BPVqueIW1D+YO9qEYFA9+Dujt0sv96/hrbj9EhEORgEwglT3m3u","p":"{\"ncbtn\":\"62|196.60000610351562|41.60000228881836|29.600000381469727|196.60000610351562|226.20000648498535|62|103.60000228881836\",\"umidToken\":\"G81C4D60AA872FBCE1E5566B72617655AA58BC074DA3F45963F\",\"ncSessionID\":\"895d44345ab\"}","scene":"register","asyn":0,"lang":"cn","v":1}',
            'x5secdata': 'xd18ce6bcd8418dcf29e0b43b00163ce296d4644747b6610d71687146106a - 717315356a683744249abaac2eaa__bx__h5api.m.tmall.com: 443 / h5 / mtop.taobao.pcdetail.data.get / 1.0',
            'landscape': 1,
            'ts': str(time.time() * 1000),
            'v': '05941877426998865'
        }
        print(data_slide)
        x5sec = '7b22617365727665723b32223a226133643033376431333131623935333230623631633235393364623938666538434a587776365147454e5867314e536d2f75693644426f4d4d6a6b794e6a41784d6a4d784d4473794d506d33684d594351414d3d227d'
        return x5secdata, x5sec
