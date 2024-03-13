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
parser.add_argument('-o', '--output', help='file name to save', type=str, default=None)
parser.add_argument('-c', '--component', help='component to select for vector-valued lineout', type=int, default=None)
parser.add_argument('-t', '--title', help='plot title', type=str, default=None)
args = parser.parse_args()

var_names = {'scalar flux': r'$\varphi$', 'current' : r'$\|J\|$'}

if (len(args.labels)):
	labels = args.labels
else:
	labels = args.files

for file,label in zip(args.files, labels):
	with open(file, 'r') as inp:
		db = yaml.safe_load(inp)

	try:
		line = db[args.key]
	except:
		s = f'key "{args.key}" not found in {file}. Valid keys are:'
		for key in list(db.keys()):
			s += f' "{key}"'
		raise RuntimeError(s)
	x = np.array(line['x'])
	y = np.array(line[args.var])
	if (len(y.shape)>1):
		if (args.component is not None):
			y = y[:,args.component]
		else:
			y = np.linalg.norm(y, axis=1)
	t = np.linalg.norm(x - x[0], axis=1)
	plt.plot(t, y, label=label)
	plt.xlabel('$t$')
	plt.ylabel(var_names[args.var])

if (len(args.files)>1):
	plt.legend()
if (args.title is not None):
	plt.title(args.title)
if (args.output is not None):
	plt.savefig(args.output)
else:
	plt.show()