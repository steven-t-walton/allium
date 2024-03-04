total = 1
scattering = .99

alpha = .25 
beta = 1
delta = 2
function scalar_flux_solution(x,y,z)
	return math.sin(math.pi*x)*math.sin(math.pi*y) + alpha*2/3*math.sin(3*math.pi*x)*math.sin(3*math.pi*y) + delta
end

function current_solution(x,y,z)
	return beta/3*math.sin(2*math.pi*x)*math.sin(2*math.pi*y)
end

function psi(x,y,z,mu,eta,xi)
	val = (math.sin(math.pi*x)*math.sin(math.pi*y) + beta*(mu+eta)*math.sin(2*math.pi*x)*math.sin(2*math.pi*y) + 
		alpha*(mu^2 + eta^2)*math.sin(3*math.pi*x)*math.sin(3*math.pi*y) + delta)/4/math.pi
	assert(val>=0)
	return val
end 

function source_function(x,y,z,mu,eta,xi)
	dpsi_dx = math.cos(math.pi*x)*math.sin(math.pi*y)/4 + beta*(mu+eta)*math.cos(2*math.pi*x)*math.sin(2*math.pi*y)/2
		+ alpha*(mu^2+eta^2)*math.cos(3*math.pi*x)*math.sin(3*math.pi*y)*3/4
	dpsi_dy = math.sin(math.pi*x)*math.cos(math.pi*y)/4 + beta*(mu+eta)*math.sin(2*math.pi*x)*math.cos(2*math.pi*y)/2 
		+ alpha*(mu^2+eta^2)*math.sin(3*math.pi*x)*math.cos(3*math.pi*y)*3/4
	return mu*dpsi_dx + eta*dpsi_dy + total*psi(x,y,z,mu,eta,xi) - scattering*scalar_flux_solution(x,y,z)/4/math.pi 
end

function inflow_function(x,y,z,mu,eta,xi)
	return psi(x,y,z,mu,eta,xi)
end

materials = {
	mat = {
		total = total, 
		scattering = scattering, 
		source = source_function
	}
}

function material_map(x,y,z) 
	return "mat" 
end 

boundary_conditions = {
	inflow = {
		type = "inflow",
		value = inflow_function
	},
	ref = {
		type = "reflective"
	}
}

function boundary_map(x,y,z)
	if (x > 0.5-1e-8) then 
		return "ref"
	else
		return "inflow"
	end
end 

Ne = 20
mesh = {
	num_elements = {Ne//2,Ne},
	extents = {0.5,1} 
}

driver = {
	fe_order = 1, 
	sn_order = 4, 
	solver = {
		type = "fp", 
		abstol = 1e-10, 
		max_iter = 50
	},
	acceleration = {
		type = "ipsmm",
		consistent = false,  
		-- bc_type = "full range", 
		solver = {
			type = "cg", 
			abstol = 1e-12,
			max_iter = 100
		}
	},
	-- preconditioner = {
	-- 	type = "p1sa", 
	-- 	solver = {
	-- 		type = "direct"
	-- 	}
	-- }
}

output = {
	paraview = "solution",
	lineout = {
		centerline = {
			from = {0,.25}, 
			to = {0.5,0.25},
			npoints = 50
		}
	},
	lineout_path = "lineout.yaml"
}