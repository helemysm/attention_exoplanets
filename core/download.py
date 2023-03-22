import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm



## build the sh file for downloading the light curves
def download_lc(catalog, output ):
    
    FLAGS_output_file = 'tess_lcs.sh'
    _wget_ = 'curl -C - -L -o '
    url_ = 'https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/'

    with open(FLAGS_output_file, "w") as f:
            f.write("#!/bin/sh\n")
            for name_file in len(files.values):

                f.write("{}{} {}{}\n".format(_wget_, name_file[2], url_, name_file[2]))

            f.write("echo 'Finish {} Tess targets to {}'\n".format(len(files.values), 'TESS/'))


catalog = 'exofop_tess_tois.csv' 
output = 'tess_data_lc/'

files = pd.read_csv(catalog)

download_lc(catalog, output)
