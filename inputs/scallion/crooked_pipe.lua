sig_max = 2000
sig_min = 0.2
pipe = {
	total = {
		type = "constant", 
		values = {sig_min}
	},
	source = 0,
	heat_capacity = 1e12,
	density = 1e-2
}

wall = {
	total = {
		type = "constant", 
		values = {sig_max},
	}, 
	source = 0,
	heat_capacity = 1e12, 
	density = 10
}

function initial_condition(x,y,z)
	return 50
end

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
		value = 500
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
	serial_refinements = 0,
	parallel_refinements = 0,
}

picard = {
	type = "picard", 
	nonlinear_solver = {
		type = "fp", 
		reltol = 1e-4, 
		max_iter = 100, 
		iterative_mode = true, 
		print_level = 0
	}, 
	energy_balance_solver = {
		type = "newton", 
		reltol = 1e-6, 
		abstol = 1e-6, 
		max_iter = 40, 
		iterative_mode = true, 
		print_level = 0
	}
}

linearized = {
	type = "linearized", 
	nonlinear_solver = {
		type = "newton", 
		reltol = 1e-4, 
		abstol = 1e-4, 
		max_iter = 1, 
		iterative_mode = true, 
		print_level = -1
	}, 
	transport_solver = {
		type = "gmres", 
		reltol = 1e-8, 
		max_iter = 100, 
		iterative_mode = false, 
		kdim = 50,
		print_level = 0,
		preconditioner = {
			type = "mip", 
			solver = {
				type = "cg", 
				reltol = 1e-10, 
				max_iter = 50, 
				iterative_mode = false, 
				print_level = 0
			}
		},
	},
	-- rebalance_solver = {
	-- 	type = "newton", 
	-- 	reltol = 1e-10, 
	-- 	abstol = 1e-10, 
	-- 	max_iter = 40,
	-- 	iterative_mode = true,
	-- 	print_level = 0
	-- },
}

driver = {
	fe_order = 1, 
	sigma_fe_order = 0, 
	sn_order = 6, 
	basis_type = "lobatto", 
	solver = linearized, 
	lump = 7, 
	fixup = {
		type = "zero and scale"
	},
	time_step = 1e-10,
	final_time = 5e-7, 
}

output = {
	root = "solution", 
	visualization = {
		type = "paraview", 
		frequency = 100
	},
	tracer = {
		locations = {
			{3.475, 1.475}, -- midpoint, last cell in thin 
			{3.475, 1.525}, -- midpoint, first cell in thick 
			{3.475, 1.275}, -- midpoint, center of pipe 
			{6.975, 0.025}, -- last cell in pipe 
		}
	}
}