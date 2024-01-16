#include "mfem.hpp"
#include "igraph.h"
#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "transport_op.hpp"
#include "mip.hpp"
#include "yaml-cpp/yaml.h"
#include "sol/sol.hpp"

int main(int argc, char *argv[]) {
	bool viz = true; 
	assert(argc==2); 

	// --- load lua file --- 
	sol::state lua; 
	lua.script_file(argv[1]); 

	// --- extract list of materials --- 
	std::vector<std::string> attr_list; 
	sol::table materials = lua["materials"]; 
	if (materials.valid()) {
		for (const auto &material : materials) {
			auto key = material.first.as<std::string>(); 
			attr_list.push_back(key); 
		}
	} else { MFEM_ABORT("materials not defined"); }

	auto nattr = attr_list.size(); 
	mfem::Vector total_list(nattr), scattering_list(nattr), source_list(nattr); 
	for (auto i=0; i<attr_list.size(); i++) {
		sol::table data = materials[attr_list[i].c_str()]; 
		total_list(i) = data["total"]; 
		scattering_list(i) = data["scattering"]; 
		source_list(i) = data["source"]; 
	}

	std::unordered_map<std::string,int> attr_map; 
	for (int i=0; i<attr_list.size(); i++) {
		attr_map[attr_list[i]] = i+1; 
	}

	// --- extract list of boundary conditions --- 
	std::vector<std::string> bdr_attr_list; 
	sol::table bcs = lua["boundary_conditions"]; 
	if (bcs.valid()) {
		for (const auto &bc : bcs) {
			auto key = bc.first.as<std::string>(); 
			bdr_attr_list.push_back(key); 
		}
	} else { MFEM_ABORT("boundary conditions not defined"); }
	auto nbattr = bdr_attr_list.size(); 
	mfem::Vector inflow_list(nbattr); 
	for (auto i=0; i<bdr_attr_list.size(); i++) {
		double data = bcs[bdr_attr_list[i].c_str()]; 
		inflow_list(i) = data; 
	}

	std::unordered_map<std::string,int> bdr_attr_map; 
	for (int i=0; i<bdr_attr_list.size(); i++) {
		bdr_attr_map[bdr_attr_list[i]] = i+1; 
	}

	// --- create coefficients for input material data ---
	mfem::PWConstCoefficient total(total_list); 
	mfem::PWConstCoefficient scattering(scattering_list); 
	mfem::PWConstCoefficient source(source_list); 
	mfem::SumCoefficient absorption(total, scattering, 1, -1); 
	mfem::PWConstCoefficient inflow(inflow_list); 

	// --- algorithmic parameters --- 
	sol::table sn = lua["sn"]; 
	int fe_order = sn["fe_order"]; 
	int sn_order = sn["sn_order"]; 
	bool dsa = sn["use_dsa"]; 
	int max_it = sn["max_it"]; 
	double tolerance = sn["tol"]; 

	// --- make mesh and solution spaces --- 
	auto mesh_node = lua["mesh"]; 
	sol::optional<std::string> fname = mesh_node["file"]; 
	mfem::Mesh mesh; 
	if (fname) {
		mesh = mfem::Mesh::LoadFromFile(fname.value(), 1, 1);
		sol::optional<int> refinements = mesh_node["refinements"]; 
		if (refinements) {
			for (int r=0; r<refinements; r++) mesh.UniformRefinement(); 			
		}
	} else {
		sol::table ne = mesh_node["num_elements"]; 
		sol::table extents = mesh_node["extents"]; 
		assert(ne.size() == extents.size()); 
		auto num_dim = ne.size(); 
		if (num_dim==1) {
			mesh = mfem::Mesh::MakeCartesian1D(ne[1], extents[1]); 
		} else if (num_dim==2) {
			mesh = mfem::Mesh::MakeCartesian2D(ne[1], ne[2], mfem::Element::QUADRILATERAL, true, extents[1], extents[2], false); 
		} else if (num_dim==3) {
			mesh = mfem::Mesh::MakeCartesian3D(ne[1], ne[2], ne[3], mfem::Element::HEXAHEDRON, extents[1], extents[2], extents[3], false); 
		} else { MFEM_ABORT("dim = " << num_dim << " not supported"); }
	}
	int dim = mesh.Dimension(); 
	std::cout << "dim = " << dim << std::endl; 

	// --- assign materials to elements --- 
	mfem::Vector center(3); 
	center = 0.0; 
	sol::function geom_func = lua["material_map"]; 
	for (int e=0; e<mesh.GetNE(); e++) {
		mfem::Vector c; 
		mesh.GetElementCenter(e, c);
		for (int d=0; d<c.Size(); d++) { center(d) = c(d); } 
		std::string attr_name = geom_func(center(0), center(1), center(2)); 
		mesh.SetAttribute(e, attr_map[attr_name]); 
	}

	// --- assign boundary conditions to boundary elements --- 
	sol::function bdr_func = lua["boundary_map"]; 
	for (int e=0; e<mesh.GetNBE(); e++) {
		const mfem::Element &el = *mesh.GetBdrElement(e); 
		int geom = mesh.GetBdrElementBaseGeometry(e);
		mfem::ElementTransformation &trans = *mesh.GetBdrElementTransformation(e); 
		mfem::Vector c(mesh.SpaceDimension()); 
		trans.Transform(mfem::Geometries.GetCenter(geom), c); 
		for (int d=0; d<c.Size(); d++) { center(d) = c(d); }
		std::string attr_name = bdr_func(center(0), center(1), center(2)); 
		mesh.SetBdrAttribute(e, bdr_attr_map[attr_name]); 
	}

	// --- build solution space --- 
	mfem::L2_FECollection fec(fe_order, dim, mfem::BasisType::GaussLegendre); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 
	auto Ndof = fes.GetVSize(); 
	auto Ne = mesh.GetNE(); 

	// --- angular quadrature rule --- 
	LevelSymmetricQuadrature quad(sn_order, dim); 
	std::cout << "num elements = " << Ne << std::endl; 
	std::cout << "SN order = " << sn_order << ", angles = " << quad.Size() << std::endl; 

	// --- setup transport vectors --- 
	TransportVectorExtents extents(1, quad.Size(), fes.GetVSize());
	auto psi_size = TotalExtent(extents); 
	std::cout << "psi size = " << psi_size << std::endl; 
	MomentVectorExtents extents_phi(1, 1, fes.GetVSize());
	auto phi_size = TotalExtent(extents_phi); 
	std::cout << "phi size = " << phi_size << std::endl;  
	DiscreteToMoment D(quad, extents, extents_phi); 

	// --- create graph --- 
	const auto &el2el = mesh.ElementToElementTable(); 
	const mfem::Table *f2el = mesh.GetFaceToElementTable(); 
	const mfem::Table *el2f = Transpose(*f2el); 

	mfem::Array<igraph_integer_t> edges; 
	edges.Reserve(el2f->Size_of_connections()*quad.Size()*2);
	mfem::Array<int> row; 
	mfem::Vector normal(dim); 
	for (int r=0; r<f2el->Size(); r++) {
		f2el->GetRow(r, row); 
		if (row.Size()==2) {
			auto ref_geom = mesh.GetFaceGeometry(r);
			const auto &int_rule = mfem::IntRules.Get(ref_geom, 1); 
			auto &trans = *mesh.GetFaceElementTransformations(r); 
			trans.SetAllIntPoints(&int_rule[0]); 
			if (dim==1) {
				normal(0) = 2*trans.GetElement1IntPoint().x - 1.0;
			} else {
				CalcOrtho(trans.Jacobian(), normal); 				
			}
			for (int a=0; a<quad.Size(); a++) {
				const auto &Omega = quad.GetOmega(a); 
				double dot = normal*Omega; 
				if (dot > 0) {
					edges.Append(row[0]+a*Ne); 
					edges.Append(row[1]+a*Ne); 
				} else {
					edges.Append(row[1]+a*Ne); 
					edges.Append(row[0]+a*Ne); 
				}
			}
		}
	}

	delete f2el; 

	// --- convert to igraph --- 
	igraph_t graph; 
	igraph_vector_int_t edges_view; 
	igraph_vector_int_view(&edges_view, edges.GetData(), edges.Size()); 
	igraph_create(&graph, &edges_view, Ne*quad.Size(), true); 

	// --- order DAG --- 
	igraph_bool_t is_dag; 
	igraph_is_dag(&graph, &is_dag); 
	if (!is_dag) { MFEM_ABORT("graph is not a dag"); }

	std::vector<igraph_integer_t> order(Ne*quad.Size()); 
	igraph_vector_int_t order_view; 
	igraph_vector_int_view(&order_view, order.data(), order.size()); 
	igraph_topological_sorting(&graph, &order_view, IGRAPH_OUT); 
	// for (int a=0; a<ordinates.size(); a++) {
	// 	std::cout << a<< ": "; 
	// 	for (int i=0; i<order.size(); i++) {
	// 		if (order[i] / Ne == a) {
	// 			std::cout << order[i] % Ne << " "; 
	// 		}
	// 	}		
	// 	std::cout << std::endl; 
	// }

	igraph_destroy(&graph); 

	// --- form isotropic source term --- 
	mfem::LinearForm bform(&fes); 
	bform.AddDomainIntegrator(new mfem::DomainLFIntegrator(source)); 
	bform.Assemble(); 

	// --- build MIP DSA operator --- 
	mfem::BilinearForm Dform(&fes); 
	mfem::RatioCoefficient diffco(1./3, total); 
	double alpha = ComputeAlpha(quad, normal)/2;
	mfem::ConstantCoefficient alpha_c(alpha); 
	Dform.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffco)); 
	Dform.AddDomainIntegrator(new mfem::MassIntegrator(absorption)); 
	Dform.AddInteriorFaceIntegrator(new MIPDiffusionIntegrator(diffco, -1, pow(fe_order+1,2), alpha)); 
	// Dform.AddBdrFaceIntegrator(new MIPDiffusionIntegrator(diffco, -1, pow(fe_order+1,2), alpha)); 
	Dform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c)); 
	Dform.Assemble(); 
	Dform.Finalize(); 

	mfem::BilinearForm Ms_form(&fes); 
	Ms_form.AddDomainIntegrator(new mfem::MassIntegrator(scattering)); 
	Ms_form.Assemble(); 
	Ms_form.Finalize(); 

	mfem::Vector scatsource(fes.GetVSize()); 
	mfem::KLUSolver solver(Dform.SpMat()); 

	mfem::Vector psi(psi_size); 
	psi = 0.0; 
	TransportVectorView psi_view(psi.GetData(), extents); 
	mfem::GridFunction phi(&fes), phi_old(&fes), delta_phi(&fes); 
	D.Mult(psi, phi); 
	phi_old = phi; 

	auto f2be = mesh.GetFaceToBdrElMap();
	for (int it=0; it<max_it; it++) {

		mfem::Array<int> faces, dofs, dofs2; 
		for (auto v : order) {
			const auto e = v % Ne; 
			const auto a = v / Ne; 
			const auto &Omega = quad.GetOmega(a); 
			const auto &el = *fes.GetFE(e); 
			auto &trans = *mesh.GetElementTransformation(e); 
			el2f->GetRow(e, faces); 
			fes.GetElementDofs(e, dofs); 

			// volumetric streaming term 
			mfem::DenseMatrix G; 
			mfem::VectorConstantCoefficient Q(Omega); 
			mfem::ConservativeConvectionIntegrator conv_int(Q, 1.0); 
			conv_int.AssembleElementMatrix(el, trans, G); 

			// collision term 
			mfem::DenseMatrix Mt; 
			mfem::MassIntegrator mt_int(total); 
			mt_int.AssembleElementMatrix(el, trans, Mt); 

			mfem::Vector rhs(dofs.Size()); 
			bform.GetSubVector(dofs, rhs); 

			// scattering 
			mfem::DenseMatrix Ms; 
			mfem::MassIntegrator ms_int(scattering); 
			ms_int.AssembleElementMatrix(el, trans, Ms); 
			mfem::Vector phi_local(dofs.Size()); 
			phi.GetSubVector(dofs, phi_local); 
			Ms.AddMult(phi_local, rhs, 1./4/M_PI); 

			mfem::DenseMatrix F(dofs.Size());
			F = 0.0; 
			double alpha = 1.0; 
			double beta = 0.5; 
			for (int fidx=0; fidx<faces.Size(); fidx++) {
				auto &face_trans = *mesh.GetFaceElementTransformations(faces[fidx]); 
				mfem::DGTraceIntegrator upw_int(Q, alpha, beta); 
				mfem::DenseMatrix f; 
				const auto &el1 = *fes.GetFE(face_trans.Elem1No); 
				const auto *el2 = &el1; 
				if (face_trans.Elem2No >= 0) {
					el2 = fes.GetFE(face_trans.Elem2No); 
				}
				int ndofs1 = el1.GetDof(); 
				int ndofs2 = el2->GetDof(); 
				upw_int.AssembleFaceMatrix(el1, *el2, face_trans, f); 
				mfem::DenseMatrix f11, f12; 
				int ep = (e==face_trans.Elem1No) ? face_trans.Elem2No : face_trans.Elem1No;
				if (e == face_trans.Elem1No) {
					f.GetSubMatrix(0, ndofs1, 0, ndofs1, f11); 
					f.GetSubMatrix(0, ndofs1, ndofs1, f.Width(), f12); 				
				} else {
					f.GetSubMatrix(ndofs1, f.Height(), ndofs1, f.Width(), f11); 
					f.GetSubMatrix(ndofs1, f.Height(), 0, ndofs1, f12); 
				}
				F += f11; 
				if (ep>=0) {
					fes.GetElementDofs(ep, dofs2); 
					mfem::Vector psi2(dofs2.Size()); 
					for (int i=0; i<dofs2.Size(); i++) { psi2(i) = psi_view(0, a, dofs2[i]); }
					f12.AddMult(psi2, rhs, -1.0); 									
				} else {
					auto be = f2be[face_trans.ElementNo]; 
					auto &bdr_face_trans = *mesh.GetBdrFaceTransformations(be); 
					mfem::ConstantCoefficient zero(1.0); 
					mfem::BoundaryFlowIntegrator bdr_flow(inflow, Q, alpha, beta);
					mfem::Vector elvec; 
					bdr_flow.AssembleRHSElementVect(el, bdr_face_trans, elvec);  
					rhs -= elvec; 
				}
			} 

			mfem::DenseMatrix A(G); 
			A += F; 
			A += Mt; 

			mfem::LinearSolve(A, rhs.GetData()); 
			for (int i=0; i<dofs.Size(); i++) { psi_view(0, a, dofs[i]) = rhs(i); }
		}

		D.Mult(psi, phi); 

		if (dsa) {
			delta_phi = phi; 
			delta_phi -= phi_old;
			Ms_form.Mult(delta_phi, scatsource);  

			solver.Mult(scatsource, delta_phi); 
			phi += delta_phi; 			
		}
		phi_old -= phi; 
		double norm = phi_old.Norml2(); 
		std::cout << "it = " << it << ", norm = " << norm << std::endl; 
		if (norm < tolerance) {
			break; 
		}
		phi_old = phi; 
	}

	if (viz) {
		mfem::ParaViewDataCollection dc("solution", &mesh); 
		dc.RegisterField("phi", &phi); 
		dc.Save(); 		
	}

	delete el2f; 
}