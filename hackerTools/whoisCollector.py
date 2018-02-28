import requests
import time
from bs4 import BeautifulSoup

startTime = time.time()

#查询域名信息
#通过http://site.ip138.com来进行查询

#http://site.ip138.com/输入你要查询的域名/domain.html #这个目录用于查询IP解析记录
#http://site.ip138.com/输入你要查询的域名/beian.html #这个用于查询子域名
#http://site.ip138.com/输入你要查询的域名/whois.html #这个用于进行whois查询
def check():
    domain = input('请输入你要查询的域名:')
    #设置User-Agent来过反爬
    header = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}

    #设置好url
    url="http://site.ip138.com/{}/".format(domain)
    urldomain="http://site.ip138.com/{}/domain.htm".format(domain)
    url2="http://site.ip138.com/{}/beian.htm".format(domain)
    url3="http://site.ip138.com/{}/whois.htm".format(domain)

    #发送请求打开网页
    rb=requests.get(url,headers=header)
    rb1=requests.get(urldomain,headers=header)
    rb2=requests.get(url2,headers=header)
    rb3=requests.get(url3,headers=header)

    #获取内容并用html的方式返回
    gf=BeautifulSoup(rb.content,'html.parser')
    print('[+]IP解析记录')
    #读取内容里的p标签
    for x in gf.find_all('p'):
        #使用text的内容返回
        link=x.get_text()
        print(link)
        
    gf1=BeautifulSoup(rb1.content,'html.parser')
    print('[+]子域名查询')
    for v in gf1.find_all('p'):
        link2=v.get_text()
        print(link2)
        
    gf2=BeautifulSoup(rb2.content,'html.parser')
    print('[+]备案查询')
    for s  in gf2.find_all('p'):
        link3=s.get_text()
        print(link3)
        
    gf3=BeautifulSoup(rb3.content,'html.parser')
    print('[+]whois查询')
    for k in gf3.find_all('p'):
        link4=k.get_text()
        print(link4)
        
check()
endTime = time.time()
print('查询耗时:',endTime-startTime)
