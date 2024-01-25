total = 1
scattering = .99

alpha = .25
delta = 2
function solution(x,y,z)
	return math.sin(math.pi*x)*math.sin(math.pi*y) + alpha*2/3*math.sin(2*math.pi*x)*math.sin(2*math.pi*y) + delta
end

function psi(x,y,z,mu,eta,xi)
	val = (math.sin(math.pi*x)*math.sin(math.pi*y) + alpha*(mu^2 + eta^2)*math.sin(2*math.pi*x)*math.sin(2*math.pi*y) + delta)/4/math.pi
	assert(val>=0)
	return val
end 

function source_function(x,y,z,mu,eta,xi)
	dpsi_dx = math.cos(math.pi*x)*math.sin(math.pi*y)/4 + alpha*(mu^2+eta^2)*math.cos(2*math.pi*x)*math.sin(2*math.pi*y)/2
	dpsi_dy = math.sin(math.pi*x)*math.cos(math.pi*y)/4 + alpha*(mu^2+eta^2)*math.sin(2*math.pi*x)*math.cos(2*math.pi*y)/2
	return mu*dpsi_dx + eta*dpsi_dy + total*psi(x,y,z,mu,eta,xi) - scattering*solution(x,y,z)/4/math.pi 
end

function inflow_function(x,y,z,mu,eta,xi)
	return psi(x,y,z,mu,eta,xi)
end

materials = {
	mat = {
		total = total, 
		scattering = scattering, 
		source = source_function
	}
}

function material_map(x,y,z) 
	return "mat" 
end 

boundary_conditions = {
	inflow = inflow_function
}

function boundary_map(x,y,z)
	return "inflow"
end 

Ne = 40
mesh = {
	num_elements = {Ne,Ne},
	extents = {1,1} 
}

sn = {
	fe_order = 1, 
	sn_order = 4, 
	acceleration = {
		type = "LDGSA", 
		solver = "cg", 
		reltol = 1e-2, 
		max_it = 50
	},
	tol = 1e-10, 
	max_it = 100, 
	solver = "gmres",
	write_graph = false
}

output = {
       name = "solution"
}