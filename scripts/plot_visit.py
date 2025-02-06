#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse 

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
	path = f'{base}_{args.ts:06d}'
	with open(f'{path}.mfem_root', 'r') as inp:
		db = json.load(inp)

	main = db['dsets']['main']
	procs = main['domains']
	time = main['time']
	time_step = main['time_step']

	x_global = []
	val_global = []
	for proc in range(procs):
		x = []
		val = []
		pid = f'{proc:06d}'
		mesh_f = open(f'{path}/mesh.{pid}', 'r')

		connectivity = []
		vertices = []

		for line in mesh_f:
			if (line.startswith('dimension')):
				dim = int(next(mesh_f))
			elif (line.startswith('elements')):
				Ne = int(next(mesh_f))
				for e in range(Ne):
					data = next(mesh_f).split(' ')
					attr = int(data[0])
					elem_type = int(data[1])
					assert(elem_type == 1)
					v1 = int(data[2])
					v2 = int(data[3])
					connectivity.append([v1,v2])
			elif (line.startswith('vertices')):
				Nvert = int(next(mesh_f))
				vdim = int(next(mesh_f))
				for v in range(Nvert):
					vertices.append(float(next(mesh_f)))

		mesh_f.close()

		T_f = open(f'{path}/{args.var}.{pid}', 'r')
		line = next(T_f)
		line = next(T_f)
		fes = line.split(': ')[1]
		line = next(T_f)
		vdim = int(line.split(': ')[1])
		line = next(T_f)
		ordering = int(line.split(': ')[1])
		line = next(T_f)

		data = []
		for line in T_f:
			data.append(float(line))

		for e in range(Ne):
			vids = connectivity[e]
			v = [vertices[i] for i in vids]
			local = [data[i] for i in [2*e, 2*e+1]]
			ips = [0,0.25,0.75,1]
			for ip in ips:
				s = np.array([1.0-ip, ip])
				x.append(v@s)
				val.append(local@s)

		x_global.append(x)
		val_global.append(val)

	X = []
	Y = [] 
	start = [x[0] for x in x_global]
	order = np.argsort(start)

	for idx in order:
		xlocal = x_global[idx]
		loc_order = np.argsort(xlocal)
		xlocal = np.array(xlocal)[loc_order]
		val_local = np.array(val_global[idx])[loc_order]
		for x,y in zip(xlocal, val_local):
			X.append(x)
			Y.append(y)

	if len(args.labels):
		plt.plot(X,Y, label=args.labels[i])
	else:
		plt.plot(X,Y)

if len(args.labels):
	plt.legend(bbox_to_anchor=(.95,0.5), loc='center left')
plt.title(r'$t =$\,' + f'{time/1e-8:.3e}' + ' sh')
if (args.x_axis is not None):
	plt.xlabel(args.x_axis)
if (args.y_axis is not None):
	plt.ylabel(args.y_axis)
if len(args.ylim):
	plt.ylim(args.ylim[0], args.ylim[1])
if (args.output is not None):
	plt.savefig(args.output, bbox_inches='tight')
else:
	plt.show()