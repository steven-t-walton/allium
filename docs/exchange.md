# Variables 
* `send_face_nbr_elements` 
	* list of local owned elements on a parallel boundary 
	* on the owned side 
	* local ids 
* `face_nbr_elements_offset`
	* offsets corresponding to each neighbor processor into `send_face_nbr_elements` 
	* counts how many elements go to each face neighbor 
* negative dofs used in RT for orientation 
* purpose of `ldof_marker`? 
* `ParMesh::GenerateOffsets` 
	* exchanges starting and ending dof offsets with neighbors 
	* accepts more than one input (e.g. dofs and true dofs) 
	* offsets are global and include a third entry that is the total number of global DOFs 

# Steps in `ExchangeFaceNbrData`
1. exchange how many DOFs exchanged for each neighbor processor 
2. exchange I arrays of sent dofs 
	* account for mismatching ordering of face neighbors using `face_nbr_elements_offset` 
3. something with `ldof_marker` to "convert" send DOFs
4. exchange J of sent dofs per element 
	* tells receiver what they are receiving 
	* uses offsets in case face neighbors not matching 
5. shift J array of reciever 
	* orders face neighbor dofs by face number? 
6. actually exchange sent DOF J's 
7. exchange DOF offset 
8. convert recieved dofs to global 
	* maps local face neighbor elements to global 