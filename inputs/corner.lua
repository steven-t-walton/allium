-- 2 material problem on 1cm x 1cm domain 

pipe = {
	total = 0.2, 
	scattering = 0, 
	source = 0
}

wall = {
	total = 200, 
	scattering = 199.9, 
	source = 0
}

materials = {pipe = pipe, wall = wall}

function material_map(x,y,z) 
	ret = "wall" 
	if (x>5/12 and x<9/12 and y>1/3) or (y>1/3 and y<2/3 and x<9/12) then 
		ret = "pipe" 
	end 
	if (x>.25 and x<1/3) and (y>5/12 and y<7/12) then 
		ret = "wall" 
	end 
	return ret 
end 

boundary_conditions = {
	inflow = {
		type = "inflow", 
		-- forward peaked inflow 
		value = function(x,y,z,mu,eta,xi)
			return mu^2
		end
	}, 
	vacuum = {
		type = "vacuum", 
	}, 
}

function boundary_map(x,y,z)
	if ((x<1e-5 and y>=1/3 and y<=2/3)) then 
		return "inflow" 
	else 
		return "vacuum" 
	end 
end 

mesh = {
	num_elements = {12,12},
	extents = {1,1}, 
	parallel_refinements = 0,
}

driver = {
	fe_order = 1, 
	sn_order = 20, 
	solver = {
		type = "gmres", 
		abstol = 1e-5, 
		max_iter = 200, 
		-- kdim = 5, 
	},
	-- acceleration = {
	-- 	type = "ldgsmm",
	-- 	consistent = false, 
	-- 	scale_stabilization = true, 
	-- 	penalty_lower_bound = true,
	-- 	solver = {
	-- 		type = "cg", 
	-- 		abstol = 1e-7, 
	-- 		max_iter = 200,
	-- 		iterative_mode = true,
	-- 	}
	-- },
	preconditioner = {
		type = "mip",
		solver = {
			type = "cg", 
			abstol = 1e-7,
			max_iter = 100,
		}
	}
}

output = {
	paraview = "solution"
}