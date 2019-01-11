"""
爬虫，从http://pds-geosciences.wustl.edu中将LIBS database扒下来

"""

import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen

html_add = "http://pds-geosciences.wustl.edu"

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

