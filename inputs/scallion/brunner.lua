mesh = {
	extents = {7,3.5},
	num_elements = {70,35}
}

sn = {
	type = "product", 
	polar_type = "legendre", 
	order = 8,
	azimuthal_order = 8,
}

energy = {
	bounds = {
		0.0,
		0.1, 
		3, 
		1.095445115010333e1, 
		4e1, 
		5e1, 
		7.825422900366437e1, 
		1.224744871391589e2, 
		1.916829312738817e2, 
		3e2, 
		6.708203932499368e2, 
		1.5e3, 
		3.240370349203930e3, 
		7e3, 
		1.114619555925213e4, 
		1.774823934729885e4, 
		2.826076380281411e4, 
		4.5e4
	},
}

materials = {
	hot_iron = {
		total = {
			type = "brunner", 
			c0 = 20.1 * 1000^3.5, 
			c1 = 1200,
			c2 = 1200,
			Nlines = 5, 
			delta_w = 10, 
			delta_s = 200, 
			Emin = 50, 
			Eedge = 7000, 
			weight = weight,
		},
		heat_capacity = 5.4273e11, 
		density = 8, 
		source = 0.0
	}, 
	cold_iron = {
		total = {
			type = "brunner", 
			c0 = 20.1 * 1000^3.5, 
			c1 = 1200,
			c2 = 1200,
			Nlines = 5, 
			delta_w = 10, 
			delta_s = 200, 
			Emin = 50, 
			Eedge = 7000, 
			weight = weight,
		},
		heat_capacity = 5.4273e11, 
		density = 6, 
		source = 0.0
	}, 
	carbon = {
		total = {
			type = "brunner", 
			c0 = 0.77 * 1000^3.5, 
			c1 = 1200,
			c2 = 30,
			Nlines = 1, 
			delta_w = 10, 
			delta_s = 1200, 
			Emin = 40, 
			Eedge = 1500, 
			weight = weight,
		},
		heat_capacity = 2.41213e11, 
		density = 2.0, 
		source = 0.0,
	},
	foam = {
		total = {
			type = "brunner", 
			c0 = 2.0 * 1000^3.5, 
			c1 = 400,
			c2 = 0,
			Nlines = 0, 
			delta_w = 0, 
			delta_s = 0, 
			Emin = 40, 
			Eedge = 300, 
			weight = weight,
		},
		density = 0.2, 
		heat_capacity = 2.41213e11, 
		source = 0.0
	}
}

function initial_condition(x,y,z)
	i = math.floor(x)
	j = math.floor(y)
	if (x>=3 and x <=4 and y>=3 and y<=4) then 
		return 500
	end 
	return 1
end

function material_map(x,y,z)
	i = math.floor(x)
	j = math.floor(y)
	if ((i == 1 or i == 5) and (j == 1 or j==5)) then 
		return "cold_iron"
	elseif (i==3 and (j==1 or j==5)) then 
		return "carbon"
	elseif ((i==2 or i==4) and (j==2 or j==4)) then 
		return "carbon"
	elseif (i==3 and j==3) then 
		return "hot_iron"
	end
	return "foam"
end

boundary_conditions = {
	hot = {
		type = "inflow", 
		value = 1000
	}, 
	cold = {
		type = "inflow", 
		value = 1
	}, 
	reflect = {
		type = "reflective"
	}
}

function boundary_map(x,y,z)
	if (x==0) then 
		return "hot"
	elseif (y==3.5) then 
		return "reflect"
	else
		return "cold"
	end
end

linearized = {
	type = "linearized", 
	-- nonlinear_solver = {
	-- 	type = "fp", 
	-- 	reltol = 1e-4, 
	-- 	max_iter = 10, 
	-- 	iterative_mode = true, 
	-- 	print_level = 1
	-- },
	rebalance_solver = {
		type = "local newton", 
		reltol = 1e-8, 
		max_iter = 40, 
		iterative_mode = true, 
		print_level = -1
	},
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
	energy_balance_solver = {
		type = "local newton", 
		reltol = 1e-11, 
		-- abstol = 0.0, 
		max_iter = 40, 
		iterative_mode = true, 
		print_level = -1
	},
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
		print_level = 0, 
	},
	ho_solver = {
		reltol = 1e-3, 
		max_iter = 40,
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

target_dt = 1e-11
initial_dt = 5e-13 
steps = 15 
driver = {
	fe_order = 1, 
	sigma_fe_order = 0, 
	gray_sigma_fe_order = 1,
	basis_type = "lobatto", 
	solver = linearized,
	lump = 7, 
	fixup = {
		type = "zero and scale", 
		psi_min = 0.0
	},
	time_step = initial_dt,
	time_step_function = function(time, dt)
		if (dt < target_dt) then 
			fac = target_dt / initial_dt
			r = math.exp(math.log(fac)/(steps-1))
			return math.min(dt*r, target_dt)
		end
		return target_dt
	end,
	final_time = 5e-8,
	-- load initial conditions from a restart file 
	-- restart = {
	--   	path = "solution/restart",
	--  	id = 0
	-- }
}

output = {
	root = "solution",
	visualization = {
		type = {"visit", "paraview"},
		times = {
			1e-10,
			2e-10,
			3e-10,
			5e-10,
			1e-9,
			2.6e-9,
			1e-8,
			1.5e-8,
			3e-8,
			4e-8
		},
	},
	restart = {
		prefix = "restart", 
		frequency = 100, -- output a restart every time step 
		num_restarts = 2 -- store 2 previous restarts 
	}, 
	tracer = {
		locations = {
			{1.5,1.5}, 
			{3.5,1.5}, 
			{5.5,1.5},
			{2.5,2.5}, 
			{4.5,2.5}, 
			{3.5,3.5},
			{3.5,0.5},
			{5.5,2.5},
			{6.5,1.5}
		},
	}, 
}
