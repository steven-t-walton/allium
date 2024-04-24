mesh = {
	num_elements = {250}, 
	extents = {1}, 
}

function initial_condition(x,y,z)
	return 1
end 

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
	if (x==0) then 
		return "inflow"
	else
		return "vacuum"
	end
end

driver = {
	fe_order = 1, 
	sigma_fe_order = 1, 
	sn_order = 6, 
	basis_type = "lobatto", 
	solver = {
		type = "gmres", 
		reltol = 1e-10, 
		max_iter = 100,
		kdim = 50,
		iterative_mode = false, 
		print_level = 0
	}, 
	newton_solver = {
		abstol = 1e-5, 
		max_iter = 1, 
	},
	lump = false, 
	sweep_opts = {
		fixup = {
			type = "zero and scale"
		}
	}, 
	time_step = 1e-10,
	final_time = 5e-8, 
	-- max_cycles = 1
}

output = {
	paraview = "solution", 
	frequency = 10
}