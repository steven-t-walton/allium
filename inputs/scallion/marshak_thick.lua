mesh = {
	num_elements = {1000}, 
	extents = {0.25}, 
}

energy = {
	num_groups = 1
}

sn = {
	type = "level symmetric", 
	order = 6
}

Tinit = 0.025
function initial_condition(x,y,z)
	return Tinit 
end 

-- sigma = alpha / T^3
materials = {
	mat = {
		total = {
			type = "analytic gray", 
			coef = 1e12, 
			nrho = 1, 
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
		type = "inflow", 
		value = Tinit
	}
}

function boundary_map(x,y,z)
	if (x==0) then 
		return "inflow"
	else
		return "vacuum"
	end
end

energy_balance_solver = {
	reltol = 1e-10, 
	max_iter = 40, 
	iterative_mode = true, 
	print_level = -1
}

picard = {
	type = "picard", 
	nonlinear_solver = {
		type = "kinsol", 
		reltol = 1e-4, 
		max_iter = 100, 
		iterative_mode = true, 
		print_level = -1,
		kdim = 0
	}, 
	energy_balance_solver = energy_balance_solver
}

linearized = {
	type = "linearized", 
	-- comment for one newton algorithm 
	nonlinear_solver = {
		type = "fp", 
		reltol = 1e-4, 
		max_iter = 10, 
		iterative_mode = true, 
		print_level = -1
	},
	rebalance_solver = energy_balance_solver,
	transport_solver = {
		type = "gmres", 
		reltol = 1e-8, 
		max_iter = 50, 
		-- max_iter = 2,
		iterative_mode = true, 
		kdim = 50,
		print_level = -1,
		preconditioner = {
			type = "mip", 
			kappa = 4,
			solver = {
				type = "cg", 
				reltol = 1e-12,
				max_iter = 50, 
				iterative_mode = false, 
				print_level = 0
			}, 
		},
	},
}

ndsa = {
	type = "ndsa", 
	lo_type = "ldg",
	outer_solver = {
		type = "fp", 
		abstol = 1e-3,
		reltol = 1e-3,
		max_iter = 50, 
		iterative_mode = true, 
	},
	energy_balance_solver = energy_balance_solver,
	inner_solver = {
		type = "fp", 
		reltol = 1e-10,
		max_iter = 40, 
		iterative_mode = true, 
	}, 
	linear_solver = {
		type = "cg", 
		reltol = 1e-11, 
		max_iter = 50, 
		iterative_mode = true, 
	},
}

smm = {
	type = "ldg", 
	consistent = true, 
	reset_to_ho = true, 
	floor_E = false, 
	opacity_integration = "outer", 
	sigmaF_weight = "rosseland",
	solver = {
		type = "cg", 
		abstol = 1e-10, 
		reltol = 1e-10, 
		max_iter = 40, 
		iterative_mode = true, 
		print_level = -1, 
	},
	ho_solver = {
		reltol = 1e-3, 
		max_iter = 40,
	},
	lo_solver = {
		reltol = 1e-3,
		max_iter = 20,
	},
	energy_balance_solver = energy_balance_solver
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
			reltol = 1e-10, 
			max_iter = 50, 
			iterative_mode = false, 
			print_level = 0
		}
	},
	energy_balance_solver = energy_balance_solver
}

driver = {
	fe_order = 1, 
	sigma_fe_order = 0, 
	basis_type = "lobatto", 
	lump = 7, 
	-- fixup = {
	-- 	type = "zero and scale", 
	-- 	psi_min = 0.0
	-- }, 
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