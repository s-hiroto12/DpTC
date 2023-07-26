

import os
from bs4 import BeautifulSoup
import requests
import urllib.request
from tqdm import tqdm
import math
import time
import re

# make dir to save
save_dir = './html/'
os.makedirs(save_dir, exist_ok=True)

# search artist
url = 'https://music.j-total.net/a_search/'
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html.parser')
print(soup)
# encodint
r.encoding = r.apparent_encoding

# get list of あ-ん
moji_links = []
gojyuon = soup.find_all('select')

for g in gojyuon:
    moji_lst = g.find_all('option')[1:]
    for moji in moji_lst:
        moji_links.append(moji.get('value'))
print(moji_links)

# for each initial letter 
artist_link_dict = {}
for moji_link in tqdm(moji_links):
    # 各文字のURL
    moji_url = url + moji_link

    # html取得
    r_moji = requests.get(moji_url)
    soup_moji = BeautifulSoup(r_moji.content, 'html.parser')

    # artist_name: url
    artist_links = soup_moji.find_all('a', href=re.compile("^//music.j-total.net/db/search.cgi"))
    for link in artist_links:
        name = link.text.replace('\n', '').replace(' ', '')
        if len(name) == 0:
            continue
        artist_link_dict[name] = 'http:' + link.get('href')
print(artist_link_dict['SaucyDog'])

print(artist_link_dict)

for artist in artist_link_dict:
    # count song number of songs
    song_cnt = 0

    # make dir to save each song
    artist_dir = save_dir + artist
    os.makedirs(artist_dir, exist_ok=True)

    # get html 
    ar_url = artist_link_dict[artist]
    try:
        r_ar = requests.get(ar_url, timeout=3.5)
    except:
        print('{} skipped'.format(artist))
        continue
    soup_ar = BeautifulSoup(r_ar.content, 'html.parser')

    # get number of songs of the artist
    pg = soup_ar.find_all('font', size='3')[-1]
    pg_string = pg.text.split(' ')
    total_num_index = pg_string.index('件中') - 1
    total_num = int(pg_string[total_num_index])
    exit()  



    # get page num
    max_page = math.ceil(total_num / 20)
    for p in range(1, max_page+1):
        page_url = ar_url + '&page={}'.format(p)
        try:
            r_song_lst = requests.get(page_url, timeout=3.5)
        except:
            print('{} {} skipped'.format(artist, p))
            continue
        soup_song_lst = BeautifulSoup(r_song_lst.content, 'html.parser')

        # get all url in the page
        song_links = soup_song_lst.find_all('a', href=re.compile("^//music.j-total.net/data"), target='')
        print(re.compile("^//music.j-total.net/db/rank.cgi\?mode"))
        print(song_links)
        # for each song 
        for s_link in song_links:
            s_url = 'http:' + s_link.get('href')
            song_name = s_link.find('b').text

            try:
                data = urllib.request.urlopen(s_url, timeout=3.5).read()
                try:
                    with open(artist_dir + '/{}.html'.format(song_name), mode='wb') as ht:
                        ht.write(data)
                except:
                    with open(artist_dir + '/{}.html'.format(song_cnt), mode='wb') as ht:
                        ht.write(data)
            except:
                print('{} {} {} skipped'.format(artist, p, s_url))
            song_cnt += 1
            time.sleep(1) 

    print('{} ok'.format(artist))

