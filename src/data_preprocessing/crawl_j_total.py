# -*- coding: utf-8 -*-
"""crawl_j_total.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DpSMBQ2xy_Q4FDKYNpkqVb3NQoTj_zR7
"""

import os
from bs4 import BeautifulSoup
import requests
import urllib.request
from tqdm import tqdm
import math
import time
import re

# 保存先作成
save_dir = './html/'
os.makedirs(save_dir, exist_ok=True)

# アーティスト検索ページ
url = 'https://music.j-total.net/a_search/'
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html.parser')
print(soup)
# エンコーディングをセット（shift-jis）
r.encoding = r.apparent_encoding

# 各五十音のリンクを取得
moji_links = []
gojyuon = soup.find_all('select')

for g in gojyuon:
    moji_lst = g.find_all('option')[1:] # 最初の要素はリンクが含まれていない
    for moji in moji_lst:
        moji_links.append(moji.get('value'))
print(moji_links)

# 頭文字ごと処理
artist_link_dict = {}
for moji_link in tqdm(moji_links):
    # 各文字のURL
    moji_url = url + moji_link

    # html取得
    r_moji = requests.get(moji_url)
    soup_moji = BeautifulSoup(r_moji.content, 'html.parser')

    # アーティストのリンクを取得して、　アーティスト名：リンク　となる辞書を作成する
    artist_links = soup_moji.find_all('a', href=re.compile("^//music.j-total.net/db/search.cgi"))
    for link in artist_links:
        name = link.text.replace('\n', '').replace(' ', '')
        if len(name) == 0:
            continue
        artist_link_dict[name] = 'http:' + link.get('href')
print(artist_link_dict['SaucyDog'])

print(artist_link_dict)

for artist in artist_link_dict:
    # 曲数をカウントする
    song_cnt = 0

    # アーティストごとにhtmlを保存するフォルダを作成する
    artist_dir = save_dir + artist
    os.makedirs(artist_dir, exist_ok=True)

    # アーティストの曲一覧ページを開いてhtml取得
    ar_url = artist_link_dict[artist]
    try:
        r_ar = requests.get(ar_url, timeout=3.5)
    except:
        print('{} skipされた'.format(artist))
        continue
    soup_ar = BeautifulSoup(r_ar.content, 'html.parser')

    # 歌手の総曲数を求める
    pg = soup_ar.find_all('font', size='3')[-1]
    pg_string = pg.text.split(' ')
    total_num_index = pg_string.index('件中') - 1
    total_num = int(pg_string[total_num_index])
    exit()  



    # 総曲数からページ数を求める
    max_page = math.ceil(total_num / 20)
    for p in range(1, max_page+1):
        page_url = ar_url + '&page={}'.format(p)
        try:
            r_song_lst = requests.get(page_url, timeout=3.5)
        except:
            print('{} {} skipされた'.format(artist, p))
            continue
        soup_song_lst = BeautifulSoup(r_song_lst.content, 'html.parser')

        # そのページの全曲URLを取得
        song_links = soup_song_lst.find_all('a', href=re.compile("^//music.j-total.net/data"), target='')
        print(re.compile("^//music.j-total.net/db/rank.cgi\?mode"))
        print(song_links)
        # 曲ごと処理
        for s_link in song_links:
            s_url = 'http:' + s_link.get('href')
            song_name = s_link.find('b').text

            # 曲名.htmlで保存したいが、曲名に変な文字が入ってる場合はsong_cntで代用する
            try:
                data = urllib.request.urlopen(s_url, timeout=3.5).read()
                try:
                    with open(artist_dir + '/{}.html'.format(song_name), mode='wb') as ht:
                        ht.write(data)
                except:
                    with open(artist_dir + '/{}.html'.format(song_cnt), mode='wb') as ht:
                        ht.write(data)
            except:
                print('{} {} {} skipされた'.format(artist, p, s_url))
            song_cnt += 1
            time.sleep(1) # お約束

    print('{} ok'.format(artist))

