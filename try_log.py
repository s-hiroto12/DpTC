
"""
print(C_major)
print(A_minor)
#chord_pitches = [pitch.Pitch(note_number) for note_number in C_major]
chord_pitches = C_major
chord_pitches2 = A_minor
chord1 = chord.Chord(chord_pitches)
chord2 = chord.Chord(chord_pitches2)
print(type(chord_pitches))
chord.duration = duration.Duration(1)

stream_obj = stream.Stream()

#measureにアペンドする方法とstreamに直接アペンドする方法がある
#コード進行だけならmeasureはいらなそう
#measure_obj = stream.Measure()
#measure_obj.append(chord)

#stream_obj.append(measure_obj)
stream_obj.append(chord1)
stream_obj.append(chord2)

stream_obj.show('midi')
"""