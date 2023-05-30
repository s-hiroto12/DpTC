from music21 import chord, instrument, note, stream, tempo

def play_chord():
    # Create a Chord object for C major
    c_major_chord = chord.Chord(["C", "E", "G"])
    
    # Set the tempo and duration
    bpm = 120
    duration = 2.0  # seconds

    # Create a Measure object
    measure = stream.Measure()
    
    # Add the chord to the measure
    measure.append(c_major_chord)
    
    # Set the instrument to Piano
    measure.insert(0, instrument.Piano())
    
    # Set the tempo for the measure
    measure.insert(0, tempo.MetronomeMark(number=bpm))
    
    # Set the duration for the measure
    measure.insert(0, note.Rest(quarterLength=duration))
    
    # Create a Stream object and add the measure
    stream_obj = stream.Stream()
    stream_obj.append(measure)
    
    # Play the stream
    stream_obj.show('midi')

# Play the C major chord
play_chord()
