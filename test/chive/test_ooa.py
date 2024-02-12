import math 
import subprocess 
import yaml 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exe', type=str, help='path to executable', required=True)
parser.add_argument('-i', '--input', type=str, help='path to base input file', required=True)
parser.add_argument('-l', '--lua', type=str, help='extra lua commands for test', required=True)
parser.add_argument('-n', '--Ne', type=int, help='number of elements in each axis for base run', default=10)
parser.add_argument('-p', '--fe_order', type=int, help='finite element order', default=1)
parser.add_argument('--phi-order', type=int, help='expected order of accuracy for phi', required=True)
parser.add_argument('--J-order', type=int, help='expected order of accuracy for J', required=True)
args = parser.parse_args()

def Run(Ne):
	cmd = ['mpirun', '-n', '4', args.exe, '-i', args.input, 
		'-l', f'driver.fe_order = {args.fe_order}; mesh.num_elements = {{{Ne},{Ne}}}; {args.lua}']
	result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
	db = yaml.safe_load(result.stdout)
	phi = db['L2 error']['scalar flux']
	J = db['L2 error']['current']
	return phi, J

err_phi1, err_J1 = Run(args.Ne)
err_phi2, err_J2 = Run(2*args.Ne)

phi_ooa = math.log2(err_phi1 / err_phi2)
print(f'phi = {phi_ooa:.3f}')
assert(math.fabs(phi_ooa - args.phi_order) < 0.2)

J_ooa = math.log2(err_J1 / err_J2)
print(f'J = {J_ooa:.3f}')
assert(math.fabs(J_ooa - args.J_order) < 0.2)