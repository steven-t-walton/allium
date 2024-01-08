epsilon = 1e-2
pipe = {
	total = 0.2, 
	scattering = 0, 
	source = 0
}

wall = {
	total = 100, 
	scattering = 99, 
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
	if (x>.4 and x<.6) and (y>.4 and y<.8) then 
		ret = "wall" 
	end 
	return ret 
end 

boundary_conditions = {
	inflow = 1, 
	vacuum = 0
}

function boundary_map(x,y,z)
	if (x<1e-5 and y>=1/3 and y<=2/3) then return "inflow" else return "vacuum" end 
end 

mesh = {
	-- file = "/opt/mfem/src/data/inline-quad.mesh",
	-- refinements = 0
	num_elements = {3,3},
	extents = {3,3} 
}

sn = {
	fe_order = 1, 
	sn_order = 2, 
	use_dsa = true,
	tol = 1e-5, 
	max_it = 50
}