pipe = {
	total = 0.2, 
	scattering = 0.2, 
	source = 0
}

wall = {
	total = 200, 
	scattering = 199.9, 
	source = 0
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
		type = "inflow", 
		value = 0 
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

Ne = 40
mesh = {
	num_elements = {7*Ne,2*Ne},
	extents = {7,2} 
}

driver = {
	fe_order = 1, 
	sn_order = 12,
	solver = {
		type = "kinsol", 
		abstol = 1e-5, 
		max_iter = 100, 
		kdim = 10, 
	},
	acceleration = {
		type = "ldgsmm",
		consistent = true, 
		scale_stabilization = true, 
		penalty_lower_bound = true,
		solver = {
			type = "cg", 
			abstol = 1e-7, 
			max_iter = 200,
			iterative_mode = true,
			amg_opts = {
				max_iter = 1, 
				pre_sweeps = 1, 
				post_sweeps = 1, 
				max_levels = 25, 
				relax_type = 8, -- 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
				aggressive_coarsening = 1, 
				interpolation = 6, -- extended+i
				coarsening = 10, -- 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
				strength_threshold = 0.25
			}
		}
	},
	-- preconditioner = {
	-- 	type = "block ldgsa",
	-- 	scale_stabilization = true, 
	-- 	solver = {
	-- 		type = "cg", 
	-- 		reltol = 1e-3, 
	-- 		-- abstol = 1e-10,
	-- 		max_iter = 100,
	-- 	}
	-- }
}

output = {
	paraview = "solution", 
	lineout = {
		{start_point = {7,0}, end_point = {7,2}, num_points = 2*Ne},
		{start_point = {0,0}, end_point = {7,0}, num_points = 7*Ne}
	}
}