$begin 'Profile'
	$begin 'ProfileGroup'
		MajorVer=2023
		MinorVer=1
		Name='Solution Process'
		$begin 'StartInfo'
			I(1, 'Start Time', '08/08/2026 00:48:30')
			I(1, 'Host', 'SCURRY')
			I(1, 'Processor', '12')
			I(1, 'OS', 'NT 10.0')
			I(1, 'Product', 'HFSS Version 2023.1.0')
		$end 'StartInfo'
		$begin 'TotalInfo'
			I(1, 'Elapsed Time', '00:02:14')
			I(1, 'ComEngine Memory', '113 M')
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
			ProfileItem('Machine', 0, 0, 0, 0, 0, 'I(5, 1, \'Name\', \'scurry\', 1, \'Memory\', \'23.6 GB\', 3, \'RAM Limit\', 90, \'%f%%\', 2, \'Cores\', 4, false, 1, \'Free Disk Space\', \'39.1 GB\')', false, true)
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
				I(1, 'Time', '08/08/2026 00:48:30')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:23')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			ProfileItem('Mesh', 0, 0, 0, 0, 41768, 'I(2, 1, \'Type\', \'Phi\', 2, \'Tetrahedra\', 2327, false)', true, true)
			ProfileItem('Post', 0, 0, 0, 0, 43316, 'I(2, 2, \'Tetrahedra\', 2346, false, 2, \'Cores\', 1, false)', true, true)
			ProfileItem('Manual Refine', 16, 0, 15, 0, 228688, 'I(3, 2, \'Tetrahedra\', 142119, false, 2, \'Cores\', 1, false, 0, \'UnifiedFeedRadiatorMesh_0p100mm\')', true, true)
			ProfileItem('Lambda Refine', 1, 0, 1, 0, 252420, 'I(2, 2, \'Tetrahedra\', 142878, false, 2, \'Cores\', 2, false)', true, true)
			ProfileItem('Simulation Setup', 1, 0, 1, 0, 323628, 'I(1, 1, \'Disk\', \'0 Bytes\')', true, true)
			ProfileItem('Port Adapt', 1, 0, 1, 0, 335704, 'I(2, 2, \'Tetrahedra\', 141851, false, 1, \'Disk\', \'136 KB\')', true, true)
			ProfileItem('Port Refine', 1, 0, 1, 0, 139636, 'I(2, 2, \'Tetrahedra\', 142997, false, 2, \'Cores\', 1, false)', true, true)
		$end 'ProfileGroup'
		$begin 'ProfileGroup'
			MajorVer=2023
			MinorVer=1
			Name='Adaptive Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '08/08/2026 00:48:54')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:01:50')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			$begin 'ProfileGroup'
				MajorVer=2023
				MinorVer=1
				Name='Adaptive Pass 1'
				$begin 'StartInfo'
					I(1, 'Frequency', '10GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 2, 0, 2, 0, 340104, 'I(2, 2, \'Tetrahedra\', 141970, false, 1, \'Disk\', \'67.5 KB\')', true, true)
				ProfileItem('Matrix Assembly', 3, 0, 9, 0, 1689064, 'I(3, 2, \'Tetrahedra\', 141970, false, 2, \'Lumped ports\', 4, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('Matrix Solve', 22, 0, 71, 0, 7141840, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 4, false, 2, \'Matrix size\', 902206, false, 3, \'Matrix bandwidth\', 22.2556, \'%5.1f\', 1, \'Disk\', \'1.56 KB\')', true, true)
				ProfileItem('Field Recovery', 2, 0, 8, 0, 7141840, 'I(2, 2, \'Excitations\', 4, false, 1, \'Disk\', \'17.8 MB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 115052, 'I(1, 0, \'Adaptive Pass 1\')', true, true)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2023
				MinorVer=1
				Name='Adaptive Pass 2'
				$begin 'StartInfo'
					I(1, 'Frequency', '10GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('Adaptive Refine', 2, 0, 2, 0, 139720, 'I(2, 2, \'Tetrahedra\', 150098, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 2, 0, 2, 0, 356092, 'I(2, 2, \'Tetrahedra\', 149068, false, 1, \'Disk\', \'73.6 KB\')', true, true)
				ProfileItem('Matrix Assembly', 3, 0, 9, 0, 1773364, 'I(3, 2, \'Tetrahedra\', 149068, false, 2, \'Lumped ports\', 4, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('Matrix Solve', 24, 0, 80, 0, 7826632, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 4, false, 2, \'Matrix size\', 947524, false, 3, \'Matrix bandwidth\', 22.2534, \'%5.1f\', 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('Field Recovery', 2, 0, 8, 0, 7826632, 'I(2, 2, \'Excitations\', 4, false, 1, \'Disk\', \'3.98 MB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 115172, 'I(1, 0, \'Adaptive Pass 2\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 3, \'Max Mag. Delta S\', 0.0308664, \'%.5f\')', false, true)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2023
				MinorVer=1
				Name='Adaptive Pass 3'
				$begin 'StartInfo'
					I(1, 'Frequency', '10GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('Adaptive Refine', 2, 0, 2, 0, 148172, 'I(2, 2, \'Tetrahedra\', 157557, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 2, 0, 2, 0, 368500, 'I(2, 2, \'Tetrahedra\', 156527, false, 1, \'Disk\', \'76.4 KB\')', true, true)
				ProfileItem('Matrix Assembly', 3, 0, 10, 0, 1857324, 'I(3, 2, \'Tetrahedra\', 156527, false, 2, \'Lumped ports\', 4, false, 1, \'Disk\', \'375 Bytes\')', true, true)
				ProfileItem('Matrix Solve', 28, 0, 92, 0, 8569640, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 4, false, 2, \'Matrix size\', 995064, false, 3, \'Matrix bandwidth\', 22.2511, \'%5.1f\', 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('Field Recovery', 2, 0, 9, 0, 8569640, 'I(2, 2, \'Excitations\', 4, false, 1, \'Disk\', \'4.32 MB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 115176, 'I(1, 0, \'Adaptive Pass 3\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 3, \'Max Mag. Delta S\', 0.0045651, \'%.5f\')', false, true)
			$end 'ProfileGroup'
			ProfileFootnote('I(1, 0, \'Adaptive Passes converged\')', 0)
		$end 'ProfileGroup'
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2023
			MinorVer=1
			Name='Simulation Summary'
			$begin 'StartInfo'
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(0, ' ')
			$end 'TotalInfo'
			GroupOptions=0
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:00\', 1, \'Total Memory\', \'108 MB\')', false, true)
			ProfileItem('Initial Meshing', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:23\', 1, \'Total Memory\', \'328 MB\')', false, true)
			ProfileItem('Adaptive Meshing', 0, 0, 0, 0, 0, 'I(5, 1, \'Elapsed Time\', \'00:01:50\', 1, \'Average memory/process\', \'8.17 GB\', 1, \'Max memory/process\', \'8.17 GB\', 2, \'Total number of processes\', 1, false, 2, \'Total number of cores\', 4, false)', false, true)
			ProfileFootnote('I(3, 2, \'Max solved tets\', 156527, false, 2, \'Max matrix size\', 995064, false, 1, \'Matrix bandwidth\', \'22.3\')', 0)
		$end 'ProfileGroup'
		ProfileFootnote('I(2, 1, \'Stop Time\', \'08/08/2026 00:50:45\', 1, \'Status\', \'Normal Completion\')', 0)
	$end 'ProfileGroup'
$end 'Profile'
