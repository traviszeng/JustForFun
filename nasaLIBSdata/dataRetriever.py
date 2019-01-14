"""
爬虫，从http://pds-geosciences.wustl.edu中将LIBS database扒下来

"""

import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import os

html_add = "http://pds-geosciences.wustl.edu"

"""
写文件
"""
def write_document(file_address,target_add):
    res = requests.get(file_address)
    if res.status_code==requests.codes.ok:
        try:
            res.raise_for_status()
        except Exception as exc:
            print('Something wrong with '+file_address)

        file =open(target_add+file_address.split('/')[-1],'wb')
        for chunk in res.iter_content(100000):
            file.write(chunk)

        file.close()
    else:
        print('network Error with'+file_address)

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
            print('Writing document'+obj['href'])
            s = obj['href'].split('/')[3:-1]
            #print(s)
            pc_add = pc_base_add
            for i in s:

                pc_add = pc_add+i
                pc_add = pc_add+'\\'
                if not os.path.exists(pc_add):
                    os.mkdir(pc_add)


            write_document(html_add+obj['href'],pc_add)


libs2_add = "http://pds-geosciences.wustl.edu/msl/msl-m-chemcam-libs-2-edr-v1/"
dir_list = []

pc_base_add = 'E:\\JustForFun\\nasaLIBSdata\\'
traverseFolder(libs2_add)
while len(dir_list) >0:
    traverseFolder(dir_list[0])
    del dir_list[0]