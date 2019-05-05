# helper.py
# Contains miscelaneous helper functions
import numpy as np
import csv

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
    with open(data_file) as csv_data:
	with open(param_file) as csv_param:

	    param_reader = csv.reader(csv_param, delimiter=',')
	    for row in param_reader:
		color = row[3]
		date = row[0]


    # From surgemes 1-3 Fill it with the first peg-number
    # for 4-5 fill it with zeros. For 6-5 fill it with
    # the mirror peg on the other side.

    pass
