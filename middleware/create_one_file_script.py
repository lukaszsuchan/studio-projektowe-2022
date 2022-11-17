file = open("file_names.txt")
data = file.read()
file_names = data.split("\n")

with open("main.py", "w+") as start_file:
    start_file.write('from collections import deque\n' +
                     'from copy import deepcopy\n' +
                     'import math\n' +
                     'import pygame\n' +
                     'import random\n' +
                     'import asyncio\n')
    for name in file_names:
        with open(name) as file:
            for line in file:
                if line.startswith('import') or line.startswith('from'): continue
                start_file.write(line)

            start_file.write("\n")