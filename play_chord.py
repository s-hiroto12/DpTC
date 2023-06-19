from chord_player import Chord_player
from music21 import *

player1 = Chord_player()

C_major = player1.construct_chord('C')
A_minor = player1.construct_chord('Am')

sample_progression = ['C', 'Am', 'C7']
player1.play_chord(sample_progression, 'Piano')
