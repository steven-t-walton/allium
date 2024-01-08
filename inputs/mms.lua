total = 1
scattering = .99

alpha = 1
delta = 2
function solution(x,y,z)
	return math.sin(math.pi*x)*math.sin(math.pi*y) + alpha*2/3*math.sin(2*math.pi*x)*math.sin(2*math.pi*y) + delta
end

function psi(x,y,z,mu,eta,xi)
	x = (math.sin(math.pi*x)*math.sin(math.pi*y) + alpha*(mu^2 + eta^2)*math.sin(2*math.pi*x)*math.sin(2*math.pi*y) + delta)/4/math.pi
	assert(x>=0)
	return x 
end 

function source_function(x,y,z,mu,eta,xi)
	dpsi_dx = math.cos(math.pi*x)*math.sin(math.pi*y)/4 + alpha*(mu^2+eta^2)*math.cos(2*math.pi*x)*math.sin(2*math.pi*y)/2
	dpsi_dy = math.sin(math.pi*x)*math.cos(math.pi*y)/4 + alpha*(mu^2+eta^2)*math.sin(2*math.pi*x)*math.cos(2*math.pi*y)/2
	return mu*dpsi_dx + eta*dpsi_dy + total*psi(x,y,z,mu,eta,xi) - scattering*solution(x,y,z)/4/math.pi 
end

materials = {
	mat = {
		total = total, 
		scattering = scattering, 
		source = 1 
	}
}

function material_map(x,y,z) 
	return "mat" 
end 

boundary_conditions = {
	vacuum = delta/4/math.pi
}

function boundary_map(x,y,z)
	return "vacuum"
end 

Ne = 10
mesh = {
	num_elements = {Ne,Ne},
	extents = {1,1} 
}

sn = {
	fe_order = 1, 
	sn_order = 4, 
	dsa = {
		kappa = 4, 
		tol = 1e-6, 
		max_it = 50
	},
	tol = 1e-10, 
	max_it = 2, 
	write_graph = false
}