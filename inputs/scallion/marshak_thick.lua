mesh = {
	num_elements = {1000}, 
	extents = {0.25}, 
}

function initial_condition(x,y,z)
	return 0.025
end 

materials = {
	mat = {
		total = {
			type = "analytic gray", 
			coef = 1e12, 
			nrho = -1, 
			nT = -3		
		}, 
		heat_capacity = 3e12, 
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
		value = 1000
	}, 
	vacuum = {
		type = "vacuum"
	}
}

function boundary_map(x,y,z)
	if (x==0) then 
		return "inflow"
	else
		return "vacuum"
	end
end

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
	-- rebalance_solver = {
	-- 	type = "newton", 
	-- 	reltol = 1e-6, 
	-- 	abstol = 1e-6, 
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
	lump = 7, 
	fixup = {
		type = "zero and scale", 
		psi_min = 0.0
	}, 
	solver = linearized, 
	time_step = 1e-12,
	final_time = 1e-8, 
}

output = {
	root = "solution", 
	visualization = {
		type = "paraview", 
		frequency = 100
	}, 
	tracer = {
		locations = {
			{0.017375}
		}
	}
}