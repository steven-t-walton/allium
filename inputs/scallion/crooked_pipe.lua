energy = {
	min = 0.0, 
	max = 1e6, 
	num_groups = 1
}

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
	heat_capacity = 1e12,
	source = 0,
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
	num_elements = {140,40},
	extents = {7,2},
	serial_refinements = 0,
	parallel_refinements = 0,
	-- partitioning = {
	-- 	type = "cartesian", 
	-- 	partitions = {8,1}
	-- }
}

sn = {
	type = "product", 
	order = 6
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
	-- nonlinear_solver = {
	-- 	type = "fp", 
	-- 	reltol = 1e-4, 
	-- 	abstol = 1e-4, 
	-- 	max_iter = 30, 
	-- 	iterative_mode = true, 
	-- 	print_level = -1
	-- }, 
	transport_solver = {
		type = "gmres", 
		reltol = 1e-8, 
		max_iter = 200, 
		iterative_mode = true, 
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
	rebalance_solver = {
		type = "local newton", 
		reltol = 1e-8, 
		max_iter = 40, 
		iterative_mode = true, 
		print_level = -1
	},
}

smm = {
	type = "ldg", 
	consistent = true, 
	reset_to_ho = false, 
	floor_E = false, 
	opacity_integration = "explicit", 
	sigmaF_weight = "rosseland",
	solver = {
		type = "cg", 
		abstol = 1e-10, 
		reltol = 1e-10, 
		max_iter = 100, 
		iterative_mode = true, 
		print_level = 0, 
	},
	ho_solver = {
		reltol = 1e-3, 
		max_iter = 100,
	},
	lo_solver = {
		reltol = 1e-3,
		max_iter = 20,
	},
	energy_balance_solver = {
		type = "local newton", 
		reltol = 1e-10, 
		-- abstol = 1e-2,
		max_iter = 40, 
		iterative_mode = true,
		print_level = -1
	}
}

teton = {
	type = "inexact newton", 
	nonlinear_solver = {
		type = "fp", 
		reltol = 1e-6, 
		max_iter = 100, 
		iterative_mode = true, 
		print_level = -1
	}, 
	preconditioner = {
		type = "mip", 
		kappa = 4, 
		solver = {
			type = "cg", 
			reltol = 1e-12, 
			max_iter = 50, 
			iterative_mode = false, 
			print_level = 0
		}
	},
	energy_balance_solver = {
		type = "local newton", 
		reltol = 1e-10, 
		-- abstol = 1e-2,
		max_iter = 40, 
		iterative_mode = true,
		print_level = -1
	}
}

driver = {
	fe_order = 1, 
	sigma_fe_order = 0, 
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
		times = {
			1e-10, 
			1e-9, 
			1e-8, 
			1e-7, 
			2e-7, 
			3e-7, 
			4e-7
		}
	},
	tracer = {
		locations = {
			{1 + 1/60, 0.0}, -- 1 cm into pipe
			{3.5 + 1/60, 1.5 - 1/60}, -- midway, last cell in pipe 
			{3.5 + 1/60, 1.5 + 1/60}, -- midway, first cell in wall 
			{6 + 1/60, 0.0}, -- 1 cm left of pipe 
			{3.0 - 1/60, 0.0}, -- last cell before blocker 
			{3.0 + 1/60, 0.0}, -- first cell in blocker 
			{4.0 - 1/60, 0.0}, -- last cell in blocker 
			{4.5 + 1/60, 0.0}, -- first cell in pipe post blocker 
		}, 
	}, 
	restart = {
		prefix = "restart", 
		frequency = 100, 
	}
}