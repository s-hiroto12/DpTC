class Note:
    name_lst = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']

    def __init__(self, s):
        """
        input
        s: string 
        name of Note
        """
        if len(s) == 1:
            self.name = s
        else:
            for index, c in enumerate(Note.name_lst):
                if len(c) != 1:
                    c_split = c.split('/')
                    if s in c_split:
                        self.name = c

    def transpose(self, step):
        """
        traspose self note step size
        step: int
        step size of transpose
        output: self

        """
        if step == 0:
            return self
        prev_index = Note.name_lst.index(self.name)
        now_index = (prev_index + step) % 12
        self.name = Note.name_lst[now_index]
        return self

    def get_interval(self, other):
        """
        return interval between 2 notes
        input
        self, other: note
        output
        int
        """
        prev_index = Note.name_lst.index(self.name)
        next_index = Note.name_lst.index(other.name)
        if prev_index > next_index:
            return next_index - prev_index - 1
        else:
            return next_index - prev_index + 1 

    def __str__(self):
        return self.name

class Chord:
    # receive chord name
    def __init__(self, s):
        """
        s:string
        name of Chord
        """
        self.name = s
        # root of code
        if len(s) == 1:
            self.root = Note(s)
        else:
            if s[1] == '#' or s[1] == 'b':
                self.root = Note(s[0:2])
            else:
                self.root = Note(s[0])

        # major or minor
        if 'm' in s:
            self.is_major = False
        else:
            self.is_major = True
    
    def transpose(self, step):
        """
        transpose chord step size
        input
        step:int
        step size of transpose
        output
        self
        """
        prev_name = self.name
        prev_root_name = self.root.name
        print('before transposed ', self.root ,self.name)
        #print(prev_name)
        #transpose root
        self.root.transpose(step)
        #change chord name
        #case: root include # or b
        if len(prev_root_name) > 2:
            prev_list = prev_root_name.split('/')
            replace_str = self.root.name
            self.name = self.root.name + self.name[2:]

        else:
            self.name = self.root.name + self.name[1:]
        print('after transposed ', self.root, self.name)
        #print(self.name)
        print()
        return self

    def __str__(self):
        if self.is_major:
            return '{}_Major'.format(self.root.name)
        else:
            return '{}_minor'.format(self.root.name)
    
class Chord_progression:
    """
    receive {key:chord_progression}
    key and chord_progression are 
    convert to
    key:Note
    chord_progressin:[Note]
    """
    def __init__(self, extracted_chord):
        extracted_key = list(extracted_chord.keys())[0]
        self.key = Note(extracted_key)
        self.chord_p = list(map(lambda s:Chord(s), extracted_chord[extracted_key]))

    def normalize(self):
        """
        transpose chord_p to C
        """
        step = self.key.get_interval(Note('C'))
        if step < 0:
            step += 1
            for chord in self.chord_p:
                chord = chord.transpose(step)
                #print(chord.name)
        else:
            step -= 1
            for chord in self.chord_p:
                chord = chord.transpose(step)
                #print(chord.name)
        return list(map(lambda c: c.name, self.chord_p))

    def __str__(self):
        chords = list(map(lambda c: c.name, self.chord_p))
        return 'key:' + self.key.name + '\nchord progression: ' + ','.join(chords)
