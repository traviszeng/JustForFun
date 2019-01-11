"""
爬虫，从http://pds-geosciences.wustl.edu中将LIBS database扒下来

"""

import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen

html_add = "http://pds-geosciences.wustl.edu"

"""
写文件
"""
def write_document(file_address):
    res = requests.get(file_address)
    if res.status_code==requests.codes.ok:
        try:
            res.raise_for_status()
        except Exception as exc:
            print('Something wrong with '+file_address)

        file =open(file_address.split('/')[-1],'wb')
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


if __name__=='__main__':
    write_document(html_add+'/msl/msl-m-chemcam-libs-2-edr-v1/mslccm_0xxx_181207.md5')

