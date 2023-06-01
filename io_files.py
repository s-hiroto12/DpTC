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
    save list as file name
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

def file_to_line(filename):
    """
    convert lines to a single line
    file_name: string
    output: string
    """
    f = get_lines(filename)
    return ' '.join(f)


    
