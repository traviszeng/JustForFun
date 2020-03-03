# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver import ChromeOptions
import os
import time
import urllib.request
import re
import getpass
import random
import platform
from os import path
import zipfile
from subprocess import run
import datetime

opt_system = ''
username = ''
password = ''
#chromedriver
driver = ''
# 不同的chrome版本对应不同的chromedriver版本
driver_version = {"69": "2.41", "68": "2.40", "67": "2.40", "66": "2.40", "65": "2.38",
                  "64": "2.37", "63": "2.36", "62": "2.34", "61": "2.33", "60": "2.33",
                  "59": "2.32", "58": "2.29", "57": "2.28", "56": "2.27", "55": "2.25",
                  "54": "2.27", "53": "2.25", "52": "2.24", "51": "2.23", "50": "2.21",
                  "49": "2.22", "48": "2.20", "47": "2.19", "46": "2.18", "45": "2.13",
                  "44": "2.19", "43": "2.17", "42": "2.15", "41": "2.13", "40": "2.12"}

#解压
def unzip(filename):
    zip_file = zipfile.ZipFile(filename)

    for names in zip_file.namelist():
        zip_file.extract(names, path.dirname(__file__))

    zip_file.close()

#登录
def login():
    # 自动下载的chromedriver的位置
    path = os.path.join(os.getcwd(),'chromedriver')

    # 添加chromeoptions
    options = ChromeOptions()
    options.add_experimental_option('detach',True)

    # 调用拉起chrome
    global driver
    driver = webdriver.Chrome(executable_path=path, options=options)
    driver.get("https://passport.weibo.cn/signin/login")
    time.sleep(1)

    driver.find_element_by_id("loginName").send_keys(username)
    time.sleep(1)
    driver.find_element_by_id("loginPassword").send_keys(password)

    time.sleep(1)
    driver.find_element_by_id("loginAction").click()
    time.sleep(3)


# 设置参数
def set_params():

    # 设置操作系统
    global opt_system
    opt_system = platform.system()

    print('请输入需要登录的用户名/手机号：')
    global username
    username = str(input())

    print('请输入密码：')
    global password
    password = str(input())

# 下载chromedriver
def download_driver():
    request = urllib.request.Request('http://npm.taobao.org/mirrors/chromedriver/')
    response = urllib.request.urlopen(request)
    html = response.read()
    r = re.compile(r'/mirrors/chromedriver/(.+?)/')
    result = r.findall(str(html))
    flag = 0

    if opt_system == 'Darwin':
        command = r'/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --version'
        s = os.popen(command).read().strip()
        version = s.split(" ")[2].split(".")

    else:
        command = r'reg query "HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Uninstall\Google Chrome" /v DisplayVersion'
        s = os.popen(command).read().strip()
        version = s.split(" ")[-1].split(".")

    if int(version[0]) <= 69:
        chrome_version = driver_version[version[0]]
    else:
        chrome_version = version[0] + '.' + version[1] + '.' + version[2]

    for item in result:
        if chrome_version in item:
            flag = item
            break

    print("正在下载驱动......")
    if opt_system == 'Darwin':
        os.system(
            "curl -O -L http://npm.taobao.org/mirrors/chromedriver/{0}/chromedriver_mac64.zip".format(
                flag))
        print("驱动下载成功")
        os.system("unzip -o chromedriver_mac64.zip")
    else:
        run("curl -O -L http://npm.taobao.org/mirrors/chromedriver/{0}/chromedriver_win32.zip".format(
            flag), shell=True)
        print("驱动下载成功")
        unzip('chromedriver_win32.zip')

    print("驱动解压成功!")

# 获取取消微博的按钮url
def get_cancel_list(driver, page_list):
    global fav_url
    cancel_list = []
    pat = re.compile(r'<div>[\u4e00-\u9fa5]+<br /><a href="(.+?)" class="cc">')
    for urlid in page_list:
        driver.get("https://weibo.cn/fav?page={0}".format(urlid))
        time.sleep(random.uniform(1, 1.5))
        result = pat.findall(driver.page_source)
        for url in result:
            cancel_list.append(url.replace("celfav", "celFavC"))
    fav_url = fav_url + cancel_list

# 删除失效的收藏
def delete_lost_favourite(count = 1):
    delete_count = 0
    driver.get("https://weibo.cn/fav?page=1")
    # pat_n是用来匹配多少页数的
    pat_n = re.compile(r'\d+/(\d+)[\u4e00-\u9fa5]</div>')

    # pat1是直接无效的
    pat1 = re.compile(r'该微博已被删除.+?/celfav/(.+?)" class="cc">取消收藏')
    # pat2是间接转发无效的
    pat2 = re.compile(r'此微博已被作者删除.+?/celfav/(.+?)" class="cc">取消收藏</a>?')

    page_list = pat_n.findall(driver.page_source)

    # 只有一页收藏的情况
    if page_list == []:
        url = "https://weibo.cn/fav?page=1"
        driver.get(url)
        result = pat1.findall(driver.page_source) + pat2.findall(driver.page_source)

        if len(result) == 0:
            return
        else:
            for url_str in result:
                driver.get("https://weibo.cn/fav/celFavC/{0}".format(url_str))
                delete_count += 1
                page_num = int(pat_n.findall(driver.page_source)[0])
                time.sleep(random.uniform(0.5, 1))
            return

    # 不止一页收藏的情况
    else:
        page_num = int(page_list[0])
    while (1):
        if int(count) <= page_num:
            url = "https://weibo.cn/fav?page={0}".format(count)
            driver.get(url)
            result = pat1.findall(driver.page_source) + pat2.findall(driver.page_source)
            if len(result) == 0:
                count += 1
                time.sleep(1.5)
                continue
            # start to delete
            for url_str in result:
                driver.get("https://weibo.cn/fav/celFavC/{0}".format(url_str))
                delete_count += 1

                time.sleep(random.uniform(0.5, 1))

            try:
                page_num = int(pat_n.findall(driver.page_source)[0])
            except Exception as e:
                print("页面刷新错误，正在重新刷新")
                time.sleep(random.uniform(1.5))
                url = "https://weibo.cn/fav?page={0}".format(count)
                driver.get(url)
                page_num = int(pat_n.findall(driver.page_source)[0])
            time.sleep(random.uniform(1, 2))
        else:
            break

    print('共清除失效收藏'+str(delete_count)+'条.')

# 删除失效转发
def delete_lost_repost(count = 1):
    delete_count = 0
    del_link = []  # 存储要删除微博的链接
    driver.get("https://weibo.cn/")
    page_code = driver.page_source
    pat = re.compile(r'<a href="/(\d+)/profile">微博\[\d*\]')
    # 找出微博id
    id = pat.findall(page_code)[0]
    time.sleep(1)
    # 微博主页
    driver.get("https://weibo.cn/{0}/profile".format(id))
    # pat_n是用来匹配多少页数的
    pat_n = re.compile(r'\d+/(\d+)[\u4e00-\u9fa5]</div>')
    page_list = pat_n.findall(driver.page_source)

    driver.get("https://weibo.cn/{0}/profile?page={1}".format(id, count))

    pat = re.compile(r'抱歉，此微博已被作者删除。查看帮助.+?<a href="(.+?)" class="cc">删除')
    pat2 = re.compile(r'.+<a href="(.+)')
    # 只有一页微博的情况
    if page_list == []:
        result = pat.findall(driver.page_source)

        for i in result:
            del_link.append(pat2.findall(i)[0])

        if len(del_link) == 0:
            return
        else:
            for url_str in del_link:
                url_list = re.split(r'[\?&]', url_str)
                link = url_list[0] + r"?type=del&" + url_list[1] + r"&act=delc&" + url_list[2] + r"&" + url_list[3]
                driver.get(link)
                delete_count += 1

                time.sleep(random.uniform(0.5, 1))
            return

    # 不止一页微博的情况
    else:
        page_num = int(page_list[0])
    while (1):
        if count <= page_num:
            driver.get("https://weibo.cn/{0}/profile?page={1}".format(id, count))
            result = pat.findall(driver.page_source)
            del_link = []
            for i in result:
                del_link.append(pat2.findall(i)[0])

            if len(del_link) == 0:
                count += 1
                time.sleep(1.5)
                continue
            for url_str in del_link:
                url_list = re.split(r'[\?&;]', url_str)
                link = url_list[0] + r"?type=del&" + url_list[1] + r"&act=delc&" + url_list[3] + r"&" + url_list[-1]
                driver.get(link)
                delete_count += 1
                time.sleep(random.uniform(0.5, 1))

        else:
            break
    print('共清除失效转发' + str(delete_count) + '条.')

# 删除所有收藏
def delete_all_favourite(count = 2):
    delete_count = 0
    driver.get("https://weibo.cn/fav?page="+str(count))
    # pat_n是用来匹配多少页数的
    pat_n = re.compile(r'\d+/(\d+)[\u4e00-\u9fa5]</div>')

    # pat1是直接无效的
    pat1 = re.compile(r'.+?/celfav/(.+?)" class="cc">取消收藏')
    # pat2是间接转发无效的
    pat2 = re.compile(r'.+?/celfav/(.+?)" class="cc">取消收藏</a>?')

    page_list = pat_n.findall(driver.page_source)

    all_result = []
    # 只有一页收藏的情况
    if page_list == []:

        url = "https://weibo.cn/fav?page=1"
        driver.get(url)
        result = pat1.findall(driver.page_source) + pat2.findall(driver.page_source)

        if len(result) == 0:
            return
        else:
            for url_str in result:
                driver.get("https://weibo.cn/fav/celFavC/{0}".format(url_str))
                delete_count += 1
                page_num = int(pat_n.findall(driver.page_source)[0])
                time.sleep(random.uniform(0.5, 1))
            return

    # 不止一页收藏的情况
    else:
        page_num = int(page_list[0])
    while (1):
        if int(count) <= page_num:
            url = "https://weibo.cn/fav?page={0}".format(count)
            print('正在清理'+url)
            driver.get(url)

            result = pat1.findall(driver.page_source) + pat2.findall(driver.page_source)
            print('获取到收藏'+str(len(result))+'条')

            count += 1
            time.sleep(1.5)
            for r in result:
                all_result.append(r)
            continue

            try:
                page_num = int(pat_n.findall(driver.page_source)[0])
            except Exception as e:
                print("页面刷新错误，正在重新刷新")
                time.sleep(random.uniform(1.5,2))
                url = "https://weibo.cn/fav?page={0}".format(count)
                driver.get(url)
                page_num = int(pat_n.findall(driver.page_source)[0])
            time.sleep(random.uniform(1, 2))

        print('开始删除')
        # start to delete
        for url_str in all_result:
            delete_flag = False
            print('正在删除：'+"https://weibo.cn/fav/celFavC/{0}".format(url_str))
            try:

                driver.get("https://weibo.cn/fav/celFavC/{0}".format(url_str))
                driver.get("https://weibo.cn/fav/celFavC/{0}".format(url_str))

            except Exception as e:
                print("页面刷新错误，正在重新刷新")

                delete_count += 1

            time.sleep(random.uniform(0.5, 1))


    print('共清除收藏'+str(delete_count)+'条.')

def main():
    # 设置参数
    set_params()

    # mac
    if opt_system == 'Darwin' and not os.path.exists(os.path.join(os.getcwd(),'chromedriver_mac64.zip')):
        download_driver()
    elif opt_system == 'Windows' and not os.path.exists(os.path.join(os.getcwd(),'chromedriver_win32.zip')):
        download_driver()

    # 设置登录账号和密码以及chromedriver
    login()

    cmd = ''
    while cmd!='c':
        cmd = str(input('清理失效收藏微博清按1\n清理失效转发微博清按2\n清理所有微博收藏请按3\n结束请按c\n'))
        if cmd == '1':
            delete_lost_favourite()
        elif cmd == '2':
            delete_lost_repost()
        elif cmd == '3':
            delete_all_favourite()


if __name__=='__main__':
    main()










