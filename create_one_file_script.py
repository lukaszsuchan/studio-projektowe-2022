file = open("file_names.txt")
data = file.read()
file_names = data.split("\n")

with open("start.py", "w+") as start_file:
    for name in file_names:
        with open(name) as file:
            for line in file:
                if line.startswith('import') or line.startswith('from .'): continue
                start_file.write(line)

            start_file.write("\n")