import sys, os
import healpy as hp, numpy as np
import os, os.path as op
import argparse
import pandas as pd
from datetime import datetime

from orphics import maps, stats, cosmology
from pixell import enmap, utils as u
from enlib import bench

import curvedsky
# add the parent dir in the python path
sys.path.append(os.path.dirname(os.getcwd()))
import param as p

parser = argparse.ArgumentParser()


