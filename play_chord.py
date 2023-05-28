from chord_player import Chord_player
from music21 import *

player1 = Chord_player()
print(player1.note_to_int)
print(player1.find_root_int('C#'))
print(player1.find_root_int('Cb'))
player1.play_chord('C')
print(player1.construct_chord('C'))

C_major = player1.construct_chord('C')
bpm = 120
duration = 2.0
measure = stream.Measure()
measure.append(C_major)
measure.insert(0, instrument.Piano())
measure.insert(0, tempo.MetronomeMark(number=bpm))
measure.insert(0, note.Rest(quarterLength=duration))
stream_obj = strem.Stream()
stream.obj.append(measure)

stream_obj.show('midi')
