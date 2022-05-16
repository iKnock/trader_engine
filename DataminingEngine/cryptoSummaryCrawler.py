# -*- coding: utf-8 -*-
"""
Created on Sat May 14 19:07:08 2022

@author: HSelato
"""
import requests
from bs4 import BeautifulSoup

import pandas as pd

num_of_crypto = 200
crypto_summary_url = "https://finance.yahoo.com/cryptocurrencies/?offset=0&count={}".format(num_of_crypto)

headers = {"User-Agent": "Chrome/96.0.4664.110"}


def GET_CRYPTO_SUMMARY_PAGE(url):
    page = requests.get(url, headers=headers)
    page_content = page.content       
    soup = BeautifulSoup(page_content, "html.parser")
    return soup



def CRYPTO_SUMMARY(soup):
    tbl = soup.find_all("table", {"class": "W(100%)"})   
    table_vals = {}
    for i in tbl:
        rows = i.find_all('tr')    
        index = 0
        for row in rows:                
            print(row.get_text(separator='|').split("|"))
            table_vals[index] =row.get_text(separator='|').split("|")
            index+=1            
    return table_vals
        
def FORMAT_CRYPTO_SUMMARY_DF(vals):
    crypto_summary_df = pd.DataFrame(vals.values())        
    #remove the last two col which are 52 week range and day chart which are none
    crypto_summary_df.drop(crypto_summary_df.columns[[-1,-2]], axis=1, inplace=True)
    
    new_header = crypto_summary_df.iloc[0] #grab the first row for the header
    crypto_summary_df = crypto_summary_df[1:] #take the data less the header row
    crypto_summary_df.columns = new_header #set the header row as the df header
    return crypto_summary_df

soup = GET_CRYPTO_SUMMARY_PAGE(crypto_summary_url)
vals = CRYPTO_SUMMARY(soup)
crypto_summery = FORMAT_CRYPTO_SUMMARY_DF(vals)