# -*- coding: utf-8 -*-


import os
import sys
import pandas as pd

path = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))

for i in range(len(labels.Class_name)):
    
    path1 = os.path.join(path,'Data','Train',labels.Class_name[i])
    path2 = os.path.join(path,'Data','Test',labels.Class_name[i])
    path3 = os.path.join(path,'Data','Val',labels.Class_name[i])
    path4 = os.path.join(path,'Raw_Data',labels.Class_name[i])
    
    path_list = [path1,path2,path3,path4]
    for dirs in path_list:
        if not os.path.exists(dirs):
            os.mkdir(dirs)
        else:
            print('\nDirectory already exists, please delete it!\n')
            sys.exit(0)
