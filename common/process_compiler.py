import re 
# 'experiments/ex4/preprocess/connect.py'
def block_str_generator(file_name):
    f = open(file_name)
    content = "".join(list(f))
    f.close()
    content = clean_up(content)
    code_lines = content.split('\n')
    blocks = generate_code_blocks(code_lines)
    cleaned_blocks = map(lambda s: re.sub(r'#<PIN>(.*)','', s), blocks)
    return cleaned_blocks 
    
def clean_up(content):
    # remove \ + newline 
    content = content.replace('\\\n', '') 
    # remove multiline comment 
    content = re.sub(
        r"'''((.|\n)*)'''", '', content)
    content = re.sub(
        r'"""((.|\n)*)"""', '', content)
    content = re.sub(r'#([^(<PIN>)|.*])\n','', content)
    content = re.sub(r'#\n','', content)
    return content

def has_last_parenthesis(code_line):
    return bool(re.search('\)(( *)|( *#<PIN>)|(#<PIN>))', code_line))

def generate_code_blocks(code_lines):
    # code_blocks = []
    tmp_str = ''
    for code_line in code_lines:
        if has_last_parenthesis(code_line): 
            tmp_str += code_line 
            # code_blocks.append(tmp_str) 
            yield tmp_str
            tmp_str = ''
        else:
            tmp_str += code_line

def handle_command(code_line):
    if '#' in code_line:
        first_command_mark_index = code_line.index('#')
        return code_line[:first_command_mark_index] 
    else:
        return code_line