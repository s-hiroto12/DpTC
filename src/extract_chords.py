import os
import re
import elements
from io_files import *

def extract_chords(filename):
    """
    extract chord progression and key from text are of tune HTML
    """
    with open(filename, 'r', encoding='UTF-8') as file:
        data = file.read()
    #check if key is informed on web site
    if '{key:' not in data:
        #if key doesnt exits, skip the tune
        return None
    data = data.split("{key:")[1:]
    sig = {'#', 'b'}
    key_chunks = []
    for key_chunk in data:
        key = None
        if key_chunk[1] in sig:
            key = key_chunk[0:2]
        else:
            key = key_chunk[0]
        # define regular expression pattern
        pattern = r"\[(.*?)\]"
        # extract str which matches the pattern
        result = re.findall(pattern, key_chunk) 
        chord_progression = []
        notes = {'A', 'B', 'C', 'D', 'E', 'F', 'G'}
        for chord in result:
            if chord[0] in notes:
                chord_progression.append(chord)
        key_chunks.append({key:chord_progression})
    return key_chunks

def split_chunks(key_chunks):
    """
    split key chunks into [{key: chord_progression}]
    chord progression is not just sequence of chord but semantically chord sequence
    input: [dict]
    output: [dict]
    """

    return None

def longest_substring(text):
    """
    return longest substring without repeating characters
    input: str 
    output: str
    """
    