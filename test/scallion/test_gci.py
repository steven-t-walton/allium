#!/usr/bin/env python3

import subprocess 
import argparse 
import yaml
import pandas as pd 
import math

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exe', type=str, help='path to executable', required=True)
parser.add_argument('-i', '--input', type=str, help='path to base input file', required=True)
parser.add_argument('--np', type=int, help='number of MPI ranks', default=1)
parser.add_argument('--dt', type=float, help='base time step', required=True)
parser.add_argument('-l', '--lua', type=str, help='extra lua commands for test', default='')
parser.add_argument('-o', '--expected_order', type=float, help='expected order of accuracy', required=True)
args = parser.parse_args()

def Run(dt):
	cmd = ['mpirun', '-n', str(args.np), args.exe, '-i', args.input, '-l', f'driver.time_step = {dt};{args.lua}']
	result = subprocess.run(cmd, stdout=subprocess.PIPE)
	output = yaml.safe_load(result.stdout)
	root = output['output']['root']
	db = pd.read_csv(f'{root}/tracer.0.csv')
	T = db['T'].iloc[-1]
	print(f'dt = {dt}, T = {T}')
	return T

f1 = Run(args.dt)
f2 = Run(args.dt/2)
f3 = Run(args.dt/4)

gci = math.log2( (f1 - f2) / (f2 - f3) )
print(f'{gci = }')
assert(math.fabs(gci - args.expected_order) < 0.3)