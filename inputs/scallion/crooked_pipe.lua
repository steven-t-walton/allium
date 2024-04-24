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

driver = {
	fe_order = 1, 
	sigma_fe_order = 0, 
	sn_order = 6, 
	basis_type = "lobatto", 
	solver = {
		type = "gmres", 
		reltol = 1e-7, 
		max_iter = 100,
		kdim = 50,
		iterative_mode = false, 
		print_level = 0
	}, 
	newton_solver = {
		abstol = 1e-4, 
		max_iter = 1, 
	},
	lump = true, 
	sweep_opts = {
		fixup = {
			type = "zero and scale", 
			psi_min = 1e-8,
		}
		-- fixup = {
			-- type = "local optimization", 
			-- psi_min = 1e-8, 
		-- }
	}, 
	time_step = 1e-10,
	final_time = 5e-7, 
	-- max_cycles = 10
}

output = {
	paraview = "solution", 
	frequency = 100
}