import os
from os.path import isfile, join
import csv
import re

data_path = './data/Yumi'
# Get all subdirectories in data_path
for dir_name, _, _ in os.walk(data_path):
    dir_files = [f for f in os.listdir(dir_name) if isfile(join(dir_name, f))]
    dir_files = [join(dir_name,f) for f in dir_files if "kinematics" in f \
            and f.endswith(".txt")]
    # Get each kinematics file
    for file_name in dir_files:
        with open(file_name,'r') as csv_original:
            with open(file_name+"replacement",'w') as csv_replacement:
                write = True
                # Add a coma in the twelfth element of each line
                reader = csv.reader(csv_original, delimiter=',')
                writer = csv.writer(csv_replacement, delimiter=',')
                for row in reader:
                    # if the file is already in the correct format, don't fix
                    if len(row)==25:
                        write = False
                        break
                    print "Fixing: ", file_name
                    new_row = row[:11]
                    # get index of problematic number
                    index = re.search('[0-9]+\.[0-9]+\-*[0-9]+\.',row[11]).end()
                    # if there is a minus, add the comma three steps before
                    if row[11][index-3] == '-':
                        index -= 3
                    # if theres is not, add the comma two steps before
                    else:
                        index -=2
                    new_row.append(row[11][0:index])
                    new_row.append(row[11][index:])
                    new_row += row[12:]
                    writer.writerow(new_row)
        if write:
            os.rename(file_name+"replacement", file_name)
        if os.path.exists(file_name+"replacement"):
            os.remove(file_name+"replacement")

