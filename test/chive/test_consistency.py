import subprocess 
import yaml 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exe', type=str, help='path to executable', required=True)
parser.add_argument('-i', '--input', type=str, help='path to base input file', required=True)
parser.add_argument('-l', '--lua', type=str, help='extra lua commands for test', required=True)
parser.add_argument('--np', type=int, help='number of MPI ranks', default=1)
args = parser.parse_args()

cmd = ['mpirun', '-n', '4', args.exe, '-i', args.input, '-l', args.lua]
result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
print(result.stdout)
db = yaml.safe_load(result.stdout)
consistency = db['consistency']
print(consistency)
assert(consistency['scalar flux'] < 1e-10)
assert(consistency['current'] < 1e-10)
