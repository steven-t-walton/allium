sig_max = 200
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
	serial_refinements = 0,
	parallel_refinements = 0,
}

sn = {
	type = "level symmetric", 
	order = 12
}

amg_opts = {
	max_iter = 1, 
	relax_sweeps = 1, 
	relax_type = 8, 
	strength_threshold = 0.25,

	-- defaults 
	-- coarsening = 10, 
	-- aggressive_coarsening = 1, 
	-- interpolation = 6, 

	-- BS recommends 
	coarsening = 6, 
	aggressive_coarsening = 0, 
	interpolation = 0
}

driver = {
	fe_order = 1, 
	basis_type = "lobatto",
	solver = {
		type = "sli", 
		abstol = 1e-6, 
		max_iter = 300, 
		kdim = 5, 
		iterative_mode = true
	},
	sweep_opts = {
	    send_buffer_size = 14
	},
	-- acceleration = {
	-- 	type = "ldgsmm",
	-- 	consistent = true, 
	-- 	bc_type = "full range",
	-- 	-- kappa = 4,
	-- 	-- scale_stabilization = true, 
	-- 	solver = {
	-- 		type = "cg", 
	-- 		abstol = 1e-8, 
	-- 		max_iter = 200,
	-- 		iterative_mode = true,
	-- 		amg_opts = amg_opts
	-- 	}
	-- },
	preconditioner = {
		type = "ldgsa",
		bc_type = "half range",
		scale_stabilization = false, 
		solver = {
			type = "cg", 
			-- reltol = 1e-3, 
			abstol = 1e-8,
			max_iter = 200,
			amg_opts = amg_opts
		}
	}
}

output = {
	paraview = "solution", 
	lineout = {
		outflow = {
			from = {7,0}, 
			to = {7,2}, 
		},
		centerline = {
			from = {0,0}, 
			to = {7,0}, 
		}, 
		inflow = {
			from = {0,0}, 
			to = {0,2}, 
		}
	},
	lineout_path = "cp_line.yaml",
	tracer = {
		{7,0}
	}
}