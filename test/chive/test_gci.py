import math
import subprocess 
import yaml 
import argparse 
from collections import defaultdict 

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exe', type=str, help='path to executable', required=True)
parser.add_argument('-i', '--input', type=str, help='path to base input file', required=True)
parser.add_argument('-l', '--lua', type=str, help='extra lua commands for test', required=True)
parser.add_argument('-p', '--fe_order', type=int, help='finite element order', default=1)
parser.add_argument('-r', '--base_ref', type=int, help='starting refinement level', default=0)
parser.add_argument('-n', '--nprocs', type=int, help='number of processors to use', default=1)
parser.add_argument('-o', '--expected_order', type=float, help='expected order of accuracy', required=True)
args = parser.parse_args()

d = []
for r in range(3):
	cmd = ['mpirun', '-n', str(args.nprocs), args.exe, '-i', args.input, 
		'-l', f'driver.fe_order = {args.fe_order}; mesh.parallel_refinements = {r+args.base_ref}; {args.lua}']
	print(*cmd)
	res = subprocess.run(cmd, stdout=subprocess.PIPE)
	db = yaml.safe_load(res.stdout)
	tracers = db['output']['tracer']
	local = []
	for tracer in tracers:
		local.append(tracer['scalar flux'])
	d.append(local)

print(d) 
at_least_one = False
for i in range(len(d[0])):
	f1 = d[0][i]
	f2 = d[1][i] 
	f3 = d[2][i]
	try:
		gci = math.log2( (f1 - f2)/(f2 - f3) )
		print(f'{gci:.3f}')
		assert(math.fabs(gci - args.expected_order) < 0.3)
		at_least_one = True
	except:
		print('log failed')

assert(at_least_one)