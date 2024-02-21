epsilon = 1e-1

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
	vacuum = {
		type = "inflow", 
		value = 0
	}
}

function boundary_map(x,y,z)
	return "vacuum"
end 

driver = {
	fe_order = 1, 
	sn_order = 4, 
	solver = {
		type = "fp", 
		abstol = 1e-10, 
		max_iter = 50, 
	},
	acceleration = {
		type = "ldgsmm", 
		consistent = true, 
		scale_stabilization = true,
		penalty_lower_bound = true,  
		solver = {
			type = "direct"
		}
	}
}
