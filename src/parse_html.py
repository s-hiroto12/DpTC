import glob
from io_files import *
import os
from bs4 import BeautifulSoup
import re
import pandas as pd
import codecs
from tqdm import tqdm

from elements import *

# regular pattern to extracting chord from html
CHORD_PATTERN = r'[A-G]{1}[#,♭]?[m]?'


def get_key_info(soup):
    """
    get key info
    input soup
    output str or None
    """
    info = soup.find_all('font', size='3')
    song_info = None

    # examine if key info is contained in html
    for i in info:
        if 'Original' in i.text:
            song_info = i.text
    # if key is not found, return None to skip this song
    if song_info is None:
        return None

    # get original key
    try:
        original_key_str = re.search(CHORD_PATTERN, song_info).group()
    except:
        return None

    return original_key_str



def parse_html(html):
    with codecs.open(html, 'r', 'shift-jis', 'ignore') as f:
        soup = BeautifulSoup(f, 'html.parser')
        song_title= soup.find('title').text.split('/')[0] # get title
        original_key = get_key_info(soup)
        if original_key == None or 'm' in original_key: # skip song whose key is empty or minor
            print(song_title + ' skipped')
            return
        original_key = original_key.replace('♭', 'b')
        chord_lst = soup.find_all('a', href=re.compile("^JavaScript:jump_1"))
        extracted_chord_progression = {original_key:[]}
        for c in chord_lst:
            # instantiation chord from chord str list
            try:
                chord_str = re.search(CHORD_PATTERN, c.text).group()
                chord_str = chord_str.replace('♭', 'b')
                extracted_chord_progression[original_key].append(chord_str)
            except:
                continue
        chord_progression = Chord_progression(extracted_chord_progression)
        normalized_progression = chord_progression.normalize()
        out_lines("data/chord_progressions/"+song_title, normalized_progression)


def extract_chords():
    artists_path = get_file_names("html")
    print(artists_path)
    for artist in artists_path:
        print('----')
        print(artist)
        html_list = get_file_names(artist)
        for html in html_list:
            parse_html(html)     
    return

extract_chords()
aggregate_files('data/chord_progressions', 'data/all_progressins')
