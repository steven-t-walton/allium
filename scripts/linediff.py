#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import yaml 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('files', help='list of files to plot', nargs=2)
parser.add_argument('-v', '--var', help='variable name', type=str, required=True)
parser.add_argument('-k', '--key', help='name of lineout to plot', type=str, required=True)
parser.add_argument('-c', '--component', help='component to select for vector-valued lineout', type=int, default=None)
args = parser.parse_args()

X = []
Y = [] 
for file in args.files:
	with open(file, 'r') as inp:
		db = yaml.safe_load(inp)

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

	X.append(x)
	Y.append(y) 

xdiff = np.linalg.norm(X[0] - X[1])
if (xdiff > 1e-14):
	raise RuntimeError('x locations not the same')

ydiff = np.linalg.norm(Y[0] - Y[1])
print(f'diff = {ydiff:.3e}')