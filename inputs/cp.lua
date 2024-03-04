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
	inflow = {
		type = "inflow", 
		value = 1.0/2/math.pi
	},
	vacuum = {
		type = "vacuum", 
	}, 
	reflection = {
		type = "reflective"
	}
}

function boundary_map(x,y,z)
	r = "vacuum" 
	if (x < 1e-10 and y < 0.5) then 
		r = "inflow"
	end 
	if (y<1e-10) then 
		r = "reflection"
	end
	return r 
end 

mesh = {
	num_elements = {7*2,2*2},
	extents = {7,2},
	serial_refinements = 0,
	parallel_refinements = 0,
}

driver = {
	fe_order = 1, 
	sn_order = 12,
	basis_type = "lobatto",
	solver = {
		type = "kinsol", 
		abstol = 1e-5, 
		max_iter = 1, 
		kdim = 10, 
		iterative_mode = true
	},
	sweep_opts = {
	    send_buffer_size = 14
	},
	acceleration = {
		type = "ldgsmm",
		consistent = true, 
		bc_type = "full range",
		-- kappa = 4,
		-- scale_stabilization = false, 
		solver = {
			type = "cg", 
			abstol = 1e-8, 
			max_iter = 200,
			iterative_mode = true,
			-- BS other options: coarsening 6, aggressive 0, interp 0
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
		}
		-- solver = {
		-- 	type = "direct", 
		-- 	iterative_refine = "none", 
		-- 	print_statistics = true
		-- }
	},
	-- preconditioner = {
	-- 	type = "mip",
	-- 	solver = {
	-- 		type = "cg", 
	-- 		-- reltol = 1e-3, 
	-- 		abstol = 1e-8,
	-- 		max_iter = 200,
	-- 	}
	-- }
}

-- output = {
-- 	paraview = "solution", 
-- 	lineout = {
-- 		outflow = {
-- 			from = {7,0}, 
-- 			to = {7,2}, 
-- 			npoints = 2*2*Ne
-- 		},
-- 		centerline = {
-- 			from = {0,0}, 
-- 			to = {7,0}, 
-- 			npoints = 2*7*Ne
-- 		}, 
-- 		inflow = {
-- 			from = {0,0}, 
-- 			to = {0,2}, 
-- 			npoints = 2*2*Ne 
-- 		}
-- 	},
-- 	lineout_path = "cp_line.yaml",
-- 	tracer = {
-- 		{2,0}, 
-- 		{7,0}
-- 	}
-- }