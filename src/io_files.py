import glob

def get_lines(filename):
    """
    return list of lines in file
    """
    f = open(filename, 'r')
    lines = f.readlines()
    new_lines = list(map(lambda s:s.rstrip('\n'), lines))
    f.close()
    return new_lines

def out_lines(filename, lines_list):
    """
    save list as filename
    """
    f = open(filename, 'w')
    for line in lines_list:
        f.write(line + '\n')
    f.close()

def get_file_names(dir):
    file_list = []
    path = dir + '/*'
    files = glob.glob(path)
    return files

def aggregate_files(input_dir, output_filename):
    """
    aggregate files in dir into a single file
    """
    files = get_file_names(input_dir)
    out_file = open(output_filename, 'w')
    for file_ in files:
        with open(file_) as f:
            progression = f.read()
            out_file.write(progression)

    out_file.close()


def file_to_line(filename):
    """
    convert lines to a single line
    file_name: string
    output: string
    """
    f = get_lines(filename)
    return ' '.join(f)


    
