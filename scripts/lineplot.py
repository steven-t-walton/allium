#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import yaml 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('files', help='list of files to plot', nargs='+')
parser.add_argument('-v', '--var', help='variable name', type=str, required=True)
parser.add_argument('-l', '--labels', help='legend entries', type=str, nargs='+', default=[])
parser.add_argument('-k', '--key', help='name of lineout to plot', type=str, required=True)
args = parser.parse_args()

var_names = {'scalar flux': r'$\varphi$', 'current' : r'$\|J\|$'}

if (len(args.labels)):
	labels = args.labels
else:
	labels = args.files

for file,label in zip(args.files, labels):
	with open(file, 'r') as inp:
		db = yaml.safe_load(inp)

	line = db[args.key]
	x = np.array(line['x'])
	y = np.array(line[args.var])
	if (len(y.shape)>1):
		y = np.linalg.norm(y, axis=1)
	t = np.linalg.norm(x - x[0], axis=1)
	plt.plot(t, y, label=label)
	plt.xlabel('$t$')
	plt.ylabel(var_names[args.var])

if (len(args.files)>1):
	plt.legend()
plt.show()