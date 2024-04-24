epsilon = 1e-1
reflect = false

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
		type = "vacuum", 
	}, 
	reflective = {
		type = "reflective",
	}
}

function boundary_map(x,y,z)
	if ((y==0.0) and reflect) then 
		return "reflective"
	else
		return "vacuum"
	end
end 

Ne = 8
L = (reflect) and 0.5 or 1.0
mesh = {
	num_elements = {Ne,Ne},
	extents = {1,L}, 
}

driver = {
	fe_order = 1, 
	sn_order = 4, 
	solver = {
		type = "fp", 
		abstol = 1e-6, 
		max_iter = 100, 
		iterative_mode = false,
		kdim = 10
	},
	-- preconditioner = {
	-- 	type = "mip",
	-- 	scale_stabilization = true,
	-- 	bc_type = "half range",
	-- }
	acceleration = {
		type = "ldgsmm", 
		bc_type = "full range", 
		consistent = true, 
		scale_stabilization = true, 
		bound_stabilization_below = true
	}
}

output = {
	paraview = "solution", 
}