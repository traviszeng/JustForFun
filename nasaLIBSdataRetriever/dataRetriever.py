"""
爬虫，从http://pds-geosciences.wustl.edu中将LIBS database扒下来

"""

import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import os
import hashlib

html_add = "http://pds-geosciences.wustl.edu"

"""
写文件

@:param
file_address:文件所在网页地址
target_folder:将文件存储到的文件夹路径

"""
def write_document(file_address,target_folder):
    res = requests.get(file_address)
    if res.status_code==requests.codes.ok:
        try:
            res.raise_for_status()
        except Exception as exc:
            print('Something wrong with '+file_address)

        file =open(target_folder+file_address.split('/')[-1],'wb')
        for chunk in res.iter_content(100000):
            file.write(chunk)

        file.close()
    else:
        print('network Error with'+file_address)

"""
解析链接，找到a标志的元素
"""
def getLink(url):
    html = urlopen(url)
    bs = BeautifulSoup(html,'html.parser')
    obj = bs.findAll('a')
    return obj


"""
遍历一个网页
"""
#write_document(html_add+'/msl/msl-m-chemcam-libs-2-edr-v1/mslccm_0xxx_181207.md5')
def traverseFolder(add):

    print('Traversing '+add)
    document_obj  = getLink(add)
    document_obj = document_obj[1:]

    for obj in document_obj:
        #如果是目录，添加到dir_list
        if obj['href'].endswith('/'):
            dir_list.append(html_add+obj['href'])
        #否则，写文件
        else:
            
            s = obj['href'].split('/')[3:-1]
            #print(s)
            add_key = ""
            pc_add = pc_base_add
            for i in s:

                pc_add = pc_add+i
                pc_add = pc_add+'\\'
                add_key+=i
                add_key+='\\'
                #add_key+='\\'
            #add_key+=obj['href'].split('/')[-1]
                if not os.path.exists(pc_add):
                    os.mkdir(pc_add)

            #pc_add ----》'E:\\JustForFun\\nasaLIBSdata\\mslccm_0xxx\\data\\sol00564\\'
            #print(add_key)
            #print(pc_add)
            #判断是否为已下载过的文件
            if s==[]:
                return
            
            filename = obj['href'].split('/')[-1]
            add_key+=filename
            #print(os.path.exists(pc_add+filename))
            #print(GetFileMD5(pc_add+filename).upper())
            #print(hash_dict[add_key])
            if os.path.exists(pc_add+filename) and GetFileMD5(pc_add+filename).upper()==hash_dict[add_key]:
                print('Found downloaded document '+obj['href'])
            else:
                write_document(html_add+obj['href'],pc_add)
                print('Writing document '+obj['href'])

#计算文件MD5
def GetFileMD5(filename):
    if not os.path.isfile(filename):
        return
    myhash = hashlib.md5()
    f = open(filename,'rb')
    while True:
        b = f.read(8096)
        if not b :
            break
        myhash.update(b)
    f.close()
    return myhash.hexdigest()

#将MD5文件读到hashmap
def readMD5File(filename):
    file = open(filename,'r')
    data = file.readlines()
    for d in data:
        md5,route = d.split()
        #print(route)
        hash_dict[route]=md5
    file.close()
        




libs2_add = "http://pds-geosciences.wustl.edu/msl/msl-m-chemcam-libs-2-edr-v1/mslccm_0xxx/calib/"
dir_list = []
hash_dict = {}
readMD5File('mslccm_0xxx_181207.md5')
readMD5File('superseded_180326.md5')

pc_base_add = 'E:\\JustForFun\\nasaLIBSdata\\'
traverseFolder(libs2_add)
while len(dir_list) >0:
    traverseFolder(dir_list[0])
    del dir_list[0]

libs2_add = "http://pds-geosciences.wustl.edu/msl/msl-m-chemcam-libs-4_5-rdr-v1/"
dir_list = []
hash_dict = {}
readMD5File('mslccm_1xxx_181207.md5')
readMD5File('superseded_181207.md5')

pc_base_add = 'E:\\JustForFun\\nasaLIBSdata2\\'
traverseFolder(libs2_add)
while len(dir_list) >0:
    traverseFolder(dir_list[0])
    del dir_list[0]

