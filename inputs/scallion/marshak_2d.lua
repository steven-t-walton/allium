mesh = {
	num_elements = {41,41},
	extents = {0.5,0.5},
}

sn = {
	type = "level symmetric", 
	order = 6
}

energy = {
	num_groups = 1
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
	-- 	type = "newton", 
	-- 	reltol = 1e-4, 
	-- 	abstol = 1e-4, 
	-- 	max_iter = 30, 
	-- 	iterative_mode = true, 
	-- 	print_level = -1
	-- }, 
	transport_solver = {
		type = "gmres", 
		reltol = 1e-6, 
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
}

driver = {
	fe_order = 1, 
	sigma_fe_order = 0, 
	basis_type = "lobatto", 
	solver = linearized, 
	lump = 7, 
	fixup = {
		type = "zero and scale", 
		psi_min = 0, 
	},
	time_step = 1e-10,
	final_time = 5e-8, 
}

output = {
	root = "solution",
	visualization = {
		type = "paraview", 
		frequency = 25, 
	}, 
	tracer = {
		locations = {
			{0.25,0.25}
		}
	}
}

materials = {
	mat = {
		total = {
			type = "analytic gray", 
			coef = 1e6, 
			nrho = -1, 
			nT = -3
		},
		heat_capacity = 1.3874e11, 
		source = 0, 
		density = 1.0
	}
}

function material_map(x,y,z)
	return "mat"
end

boundary_conditions = {
	inflow = {
		type = "inflow", 
		value = 150
	}, 
	vacuum = {
		type = "vacuum"
	}
}

function boundary_map(x,y,z)
	if (y==0) then 
		return "inflow"
	else
		return "vacuum"
	end
end

function initial_condition(x,y,z)
	return 0.025
end 