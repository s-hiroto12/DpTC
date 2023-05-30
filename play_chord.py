from chord_player import Chord_player
from music21 import *

player1 = Chord_player()
print(player1.note_to_int)
print(player1.find_root_int('C#'))
print(player1.find_root_int('Cb'))
player1.play_chord('C')
print(player1.construct_chord('C'))

C_major = player1.construct_chord('C')

print(C_major)
#chord_pitches = [pitch.Pitch(note_number) for note_number in C_major]
chord_pitches = C_major
chord = chord.Chord(chord_pitches)
print(type(chord_pitches))
chord.duration = duration.Duration(1)

stream_obj = stream.Stream()

#measureにアペンドする方法とstreamに直接アペンドする方法がある
#コード進行だけならmeasure入らなそう
#measure_obj = stream.Measure()
#measure_obj.append(chord)

#stream_obj.append(measure_obj)
stream_obj.append(chord)

stream_obj.show('midi')