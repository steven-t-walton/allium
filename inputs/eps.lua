epsilon = 1e-4

mat = {
	total = 1/epsilon, 
	scattering = 1/epsilon - epsilon,
	source = epsilon
}

materials = {mat = mat}

function material_map(x,y,z) 
	return "mat" 
end 

boundary_conditions = {
	vacuum = 0
}

function boundary_map(x,y,z)
	return "vacuum"
end 

Ne = 20
mesh = {
	-- file = "/opt/mfem/data/star-hilbert.mesh",
	-- refinements = 3
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
	tol = 1e-5, 
	max_it = 100, 
	write_graph = false
}