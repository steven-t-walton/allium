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
	-- if (y>1/3 and y<2/3 and x<1/3) then return "pipe" 
	-- elseif (x>1/3 and x<2/3 and y>2/3) then return "pipe" 
	-- elseif (x>1/3 and x<2/3 and y>1/3 and y<2/3) then return "pipe"
	-- else return "wall"
	-- end 
	ret = "wall" 
	if (x>1/3 and x<2/3 and y>1/3) or (y>1/3 and y<2/3 and x<2/3) then 
		ret = "pipe" 
	end 
	if (x>1/3-.1 and x<1/3) and (y>.45 and y<.55) then 
		ret = "wall" 
	end 
	return ret 
end 

boundary_conditions = {
	inflow = 1, 
	vacuum = 0
}

function boundary_map(x,y,z)
	-- if ((x<1e-5 and y>=1/3 and y<=2/3) or (x>=1/3 and x<=2/3 and y>1-1e-6)) then return "inflow" else return "vacuum" end 
	if ((x<1e-5 and y>=1/3 and y<=2/3)) then return "inflow" else return "vacuum" end 
end 

Ne = 80
mesh = {
	-- file = "/opt/mfem/data/inline-quad.mesh",
	-- refinements = 3
	num_elements = {Ne,Ne},
	extents = {1,1} 
}

driver = {
	fe_order = 1, 
	sn_order = 16, 
	solver = {
		type = "gmres", 
		abstol = 1e-5, 
		max_iter = 200, 
	},
	-- acceleration = {
	-- 	type = "LDGSMM",
	-- 	solver = {
	-- 		type = "direct", 
	-- 	}
	-- },
	preconditioner = {
		type = "ldgsa",
		solver = {
			type = "cg", 
			reltol = 1e-2, 
			max_iter = 100,
		}
	}
}

output = {
	name = "solution"
}