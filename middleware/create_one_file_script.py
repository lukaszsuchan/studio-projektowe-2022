file = open("file_names.txt")
data = file.read()
file_names = data.split("\n")

with open("start.py", "w+") as start_file:
    start_file.write('from collections import deque\n' +
                     'from copy import deepcopy\n'
                     'from scipy.spatial import distance\n' +
                     'import pygame\n' +
                     'import numpy as np\n' +
                     'from numpy.random import randint\n' +
                     'from pygame import gfxdraw\n')
    for name in file_names:
        with open(name) as file:
            for line in file:
                if line.startswith('import') or line.startswith('from'): continue
                start_file.write(line)

            start_file.write("\n")