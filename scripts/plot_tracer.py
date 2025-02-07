#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse 

def Load(file, key):
	df = pd.read_csv(file)
	return np.array(df['time']), np.array(df[key])

parser = argparse.ArgumentParser()
parser.add_argument('bases', help='list of files to plot', nargs='+')
parser.add_argument('-s', '--ts', help='time step id', type=int, default=1)
parser.add_argument('-v', '--var', help='variable name', type=str, required=True)
parser.add_argument('-l', '--labels', help='legend entries', type=str, nargs='+', default=[])
parser.add_argument('-o', '--output', help='file name to save', type=str, default=None)
parser.add_argument('-x', '--x-axis', help='label for x axis', type=str, default=None)
parser.add_argument('-y', '--y-axis', help='label for y axis', type=str, default=None)
parser.add_argument('-ylim', '--ylim', help='ylim', nargs=2, type=float, default=[])
# parser.add_argument('-c', '--component', help='component to select for vector-valued lineout', type=int, default=None)
# parser.add_argument('-t', '--title', help='plot title', type=str, default=None)
# parser.add_argument('-lfs', '--label-font-size', help='font size for axis labels', default=18, type=int)
# parser.add_argument('-legfs', '--legend-font-size', help='font size for legend entries', default=18, type=int)
args = parser.parse_args()

for i,base in enumerate(args.bases):
	t,v = Load(base, args.var)
	plt.semilogx(t,v)
plt.show()