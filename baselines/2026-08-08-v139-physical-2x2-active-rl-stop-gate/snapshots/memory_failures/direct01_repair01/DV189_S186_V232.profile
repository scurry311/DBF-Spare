$begin 'Profile'
	$begin 'ProfileGroup'
		MajorVer=2023
		MinorVer=1
		Name='Solution Process'
		$begin 'StartInfo'
			I(1, 'Start Time', '08/08/2026 00:38:20')
			I(1, 'Host', 'SCURRY')
			I(1, 'Processor', '12')
			I(1, 'OS', 'NT 10.0')
			I(1, 'Product', 'HFSS Version 2023.1.0')
		$end 'StartInfo'
		$begin 'TotalInfo'
		$end 'TotalInfo'
		GroupOptions=8
		TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Executing From\', \'D:\\\\v231\\\\Win64\\\\HFSSCOMENGINE.exe\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2023
			MinorVer=1
			Name='HPC'
			$begin 'StartInfo'
				I(1, 'Type', 'Auto')
				I(1, 'MPI Vendor', 'Intel')
				I(1, 'MPI Version', '2018')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(0, ' ')
			$end 'TotalInfo'
			GroupOptions=0
			TaskDataOptions(Memory=8)
			ProfileItem('Machine', 0, 0, 0, 0, 0, 'I(5, 1, \'Name\', \'scurry\', 1, \'Memory\', \'23.6 GB\', 3, \'RAM Limit\', 90, \'%f%%\', 2, \'Cores\', 4, false, 1, \'Free Disk Space\', \'39.2 GB\')', false, true)
		$end 'ProfileGroup'
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Allow off core\', \'True\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Solution Basis Order\', \'1\')', false, true)
		ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(1, 0, \'Elapsed time : 00:00:00 , HFSS ComEngine Memory : 108 M\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Perform full validations with standard port validations\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2023
			MinorVer=1
			Name='Initial Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '08/08/2026 00:38:20')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:01:19')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			ProfileItem('Mesh', 0, 0, 0, 0, 41496, 'I(2, 1, \'Type\', \'Phi\', 2, \'Tetrahedra\', 2327, false)', true, true)
			ProfileItem('Post', 0, 0, 0, 0, 43084, 'I(2, 2, \'Tetrahedra\', 2346, false, 2, \'Cores\', 1, false)', true, true)
			ProfileItem('Manual Refine', 57, 0, 56, 0, 674188, 'I(3, 2, \'Tetrahedra\', 465677, false, 2, \'Cores\', 1, false, 0, \'UnifiedFeedRadiatorMesh_0p100mm\')', true, true)
			ProfileItem('Lambda Refine', 4, 0, 4, 0, 713308, 'I(2, 2, \'Tetrahedra\', 466251, false, 2, \'Cores\', 2, false)', true, true)
			ProfileItem('Simulation Setup', 6, 0, 6, 0, 977680, 'I(1, 1, \'Disk\', \'0 Bytes\')', true, true)
			ProfileItem('Port Adapt', 2, 0, 2, 0, 977680, 'I(2, 2, \'Tetrahedra\', 446711, false, 1, \'Disk\', \'171 KB\')', true, true)
			ProfileItem('Port Refine', 5, 0, 5, 0, 410800, 'I(2, 2, \'Tetrahedra\', 466519, false, 2, \'Cores\', 1, false)', true, true)
		$end 'ProfileGroup'
	$end 'ProfileGroup'
$end 'Profile'
