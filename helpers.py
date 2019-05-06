# helper.py
# Contains miscelaneous helper functions
import numpy as np
import csv
import ast
import os
from os.path import isfile, join

################################
# Data parser functions
# inputs:
    # data_file: if its a file, just process that file.
          # in that directory
    # param_file: task parameters path file
    # recursive: if True, process every subdirectory
               # inside the directory
# output:
    # a n by m numpy array, where n is the sample number
    # and m is made of [tayectory_features, label,
    # target peg num, rotation type]
    # if the surgeme does not have a target peg/pole
    # (the case for get-together/exchagnge) the number
    # will be filled with zero.
    #################################################
    # Peg numbers are annotated like this (For Yumi):
    # (Human Left)                     (Human Right)
          # 10                               4
      # 9         11                     3         5
      # 8         12                     2         6
           # 7                                1

def parse_input_data(data_file, param_file, recursive=False):
    output = []
    # Get the data in the params file
    with open(param_file) as csv_param:
        param_reader = csv.reader(csv_param, delimiter=',')
        param_rows = [row for row in param_reader]

    file_list = []
    if os.path.isfile(data_file):
       file_list.append(data_file)
    elif recursive:
        for dir_name, _, _ in os.walk(data_file):
            dir_files = [f for f in os.listdir(dir_name) if isfile(join(dir_name, f))]
            file_list = [join(dir_name,f) for f in dir_files if "kinematics" in f \
                    and f.endswith(".txt")]
    else:
        dir_files = [f for f in os.listdir(data_file) if isfile(join(data_file, f))]
        file_list = [join(data_file,f) for f in dir_files if  "kinematics" in f \
                    and f.endswith(".txt")]

    for file_name in file_list:
        with open(file_name) as csv_data:
            data_reader = csv.reader(csv_data, delimiter=',')
            output_row = []
            transfer_step_complete = False
            peg_index = 0
            # Filter out the param for the task
            task_params = None
            for first in data_reader:
                print file_name
                subject,trial = first[0].split("_")
                task_params = [p for p in  param_rows if \
                        p[0]==subject and p[1]==trial][0]
                print task_params[3]
                task_params[3] = ast.literal_eval(task_params[3])
                print task_params[3]
                break
            # reset csv so we can look at it from the begining
            csv_data.seek(0)
            for data_row in data_reader:
                # if there is no label, skip reading
                if data_row[22] == 'False':
                    continue
                # get the traslation, rotation
                output_row += map(float,data_row[9:21])
                # add the label as an int
                output_row.append(int(data_row[23][1]))
                # Check if current surgeme is S1, if S7 happened,
                # add 1 to the counter.  and reset the S7 checker.
                current_surgeme = output_row[-1]
                if current_surgeme == 1 and transfer_step_complete:
                    peg_index += 1
                    transfer_step_complete = False
                elif current_surgeme == 7:
                    transfer_step_complete = True
                # From surgemes 1-3 Fill it with the first peg-number
                # for 4-5 fill it with zeros. For 6-5 fill it with
                # the mirror peg on the other side.
                if current_surgeme < 4:
                    peg_offset = 6 if task_params[2] == 'L-R' else 0
                    peg_num = task_params[3][peg_index]+peg_offset
                elif current_surgeme < 6:
                    peg_num = 0
                else:
                    peg_offset = 0 if task_params[2] == 'L-R' else 6
                    peg_num = task_params[3][peg_index]+peg_offset
                output_row.append(peg_num)
                output_row.append(int(task_params[-1]))
                output.append(output_row)
    return output


print parse_input_data('./data/Yumi','./data/Yumi/new_task_params.csv',True)
