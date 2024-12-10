sig_max = 2
sig_min = 0.2
pipe = {
	total = sig_min, 
	scattering = sig_min - 1e-3, 
	source = 1e-7
}

wall = {
	total = sig_max, 
	scattering = sig_max - 1e-3, 
	source = 1e-7
}

materials = {pipe = pipe, wall = wall}

function material_map(x,y,z) 
	r = "wall"
	if (y<0.5) then 
		r = "pipe"
	end
	if (x > 2.5 and x < 4.5 and y<1.5) then 
		r = "pipe"
	end
	if (x > 3 and x < 4 and y < 1) then
		r = "wall"
	end 
	return r 
end 

boundary_conditions = {
	pipe = {
		type = "inflow", 
		value = 1.0/2/math.pi
	},
	wall = {
		type = "vacuum", 
	}, 
	bottom = {
		type = "reflective"
	}
}

function boundary_map(x,y,z)
	r = "wall" 
	if (x < 1e-10 and y < 0.5) then 
		r = "pipe"
	end 
	if (y<1e-10) then 
		r = "bottom"
	end
	return r 
end 

mesh = {
	num_elements = {14,4},
	extents = {7,2},
	sfc_ordering = false,
	serial_refinements = 0,
	parallel_refinements = 0,
}

cg_solver = {
	type = "cg", 
	abstol = 1e-7, 
	max_iter = 100, 
	iterative_mode = false,
}

sn = {
	type = "level symmetric", 
	order = 8
}

driver = {
	fe_order = 1, 
	basis_type = "lobatto",
	solver = {
		type = "gmres", 
		abstol = 1e-5, 
		max_iter = 300, 
		kdim = 10, 
		iterative_mode = false
	},
}

output = {
	tracer = {
		{7,0}, 
	}
}