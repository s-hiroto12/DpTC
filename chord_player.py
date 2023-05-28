from music21 import *

class Chord_player:
    """
    Chord_player has {Note name : music21.Note.curnote}
    """
    def __init__(self):
        self.note_to_int = {
            'C':60, 
            'D':62,
            'E':64,
            'F':65,
            'G':67,
            'A':69,
            'B':71,
        }

    def play_chord(self, chord_progression):
        """
        play chord progression
        input [str]
        output None
        """
        for chord_name in chord_progression:
            self.chord_to_midi()

    def find_root_int(self, chord_name):
        """
        find root note of chord name
        input str
        output int
        """
        if '#' in chord_name:
            return self.note_to_int[chord_name[0]] + 1
        elif 'b' in chord_name:
            return self.note_to_int[chord_name[0]] - 1
        else:
            return self.note_to_int[chord_name[0]]

    
    def construct_chord(self, chord_name):
        """
        construct chord int 
        input str
        output [int] chord_int
        """
        chord_constructer = []
        root_int = self.find_root_int(chord_name)
        chord_constructer.append(root_int)
        # first, check if chord is on chord or not.
        """
        on chord: implement later
        if '/' in chord_name:
            lowest_note = chord_name.split('/')[1]
            lowest_note_int = find_root_int(lowest_note)
            return 
        """

        # not on chord
        if 'm' in chord_name and 'di' not in chord_name: # minor
            chord_constructer.append(root_int+3)
            chord_constructer.append(root_int+7)
            if '7' in chord_name: # minor 7
                chord_constructer.append(root_int+10)
                return chord_constructer
        elif 'sus' in chord_name: # sus4
            return 
        elif 'dim' in chord_name: # dim
            return 
        else: # major
            chord_constructer.append(root_int+4)
            chord_constructer.append(root_int+7)
            if '7' in chord_name:
                if 'M' in chord_name: # M7
                    return chord_constructer.append(root_int+11)
                else: # 7
                    return chord_constructer.append(root_int+10)
        return chord_constructer

    
    

    def chord_to_midi(self):
        """
        translate chord name into note(music21)
        input str
        output music21.Note
        """
        return None