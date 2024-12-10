mesh = {
	num_elements = {256}, 
	extents = {4}, 
}

sn = {
	type = "level symmetric", 
	order = 6
}

energy = {
	min = 1e-2, 
	max = 1e6, 
	spacing = "log", 
	num_groups = 20,
	extend_to_zero = true, -- add uniform spaced group from [0,min]
	print_bounds = true -- print group structure to YAML 
}

function initial_condition(x,y,z)
	return 1
end 

cv = 5.109e11
materials = {
	thin = {
		total = {
			type = "analytic", 
			coef = 1e9, 
			nrho = -1, 
			nT = -3		
		}, 
		heat_capacity = cv, 
		source = 0, 
		density = 1.0
	},
	thick = {
		total = {
			type = "analytic", 
			coef = 1e12, 
			nrho = -1, 
			nT = -3		
		}, 
		heat_capacity = cv, 
		source = 0, 
		density = 1.0
	}
}

function material_map(x,y,z)
	if (x < 1) then
		return "thin"
	elseif (x < 2) then 
		return "thick"
	else 
		return "thin"
	end
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
	if (x==4.0) then 
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
		abstol = 1e-4, 
		max_iter = 500, 
		iterative_mode = true, 
		print_level = 0,
		kdim = 0
	}, 
	energy_balance_solver = {
		type = "local newton", 
		reltol = 1e-8, 
		-- abstol = 1e-2,
		max_iter = 40, 
		iterative_mode = true,
		print_level = -1
	}
}

linearized = {
	type = "linearized", 
	transport_solver = {
		type = "gmres", 
		abstol = 1e-6,
		reltol = 1e-12, 
		max_iter = 500, 
		iterative_mode = true, 
		kdim = 50,
		print_level = 0,
		preconditioner = {
			type = "mip", 
			kappa = 4,
			solver = {
				type = "cg", 
				abstol = 1e-7, 
				reltol = 1e-8,
				max_iter = 50, 
				iterative_mode = false, 
				print_level = 0
			}, 
		},
	},
}

driver = {
	fe_order = 1, 
	sigma_fe_order = 0, 
	gray_sigma_fe_order = 1,
	basis_type = "lobatto", 
	solver = linearized,
	lump = 7, 
	-- fixup = {
	-- 	type = "zero and scale", 
	-- 	psi_min = 0.0
	-- },
	time_step = 1e-11,
	final_time = 1e-8, 
	-- load initial conditions from a restart file 
	-- restart = {
	-- 	path = "larsen/restart",
	-- 	id = 0
	-- }
}

output = {
	root = "solution", 
	visualization = {
		type = "paraview", 
		frequency = 10, 
		-- have paraview continue previous run 
		-- when restart is used 
		-- restart_mode = true, 	
	},
	tracer = {
		locations = {
			{1.7421875}, 
			{0.8515625}
		}
	}, 
	restart = {
		prefix = "restart", 
		frequency = 1, -- output a restart every time step 
		num_restarts = 2 -- store 2 previous restarts 
	}
}