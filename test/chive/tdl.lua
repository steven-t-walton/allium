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

mesh = {
	num_elements = {8,8}, 
	extents = {1,1}
}

driver = {
	fe_order = 1, 
	sn_order = 4, 
	solver = {
		type = "sli", 
		abstol = 1e-6, 
		max_iter = 50, 
	},
}
