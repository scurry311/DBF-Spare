$begin 'Profile'
	$begin 'ProfileGroup'
		MajorVer=2023
		MinorVer=1
		Name='Solution Process'
		$begin 'StartInfo'
			I(1, 'Start Time', '07/21/2026 10:10:52')
			I(1, 'Host', 'SCURRY')
			I(1, 'Processor', '12')
			I(1, 'OS', 'NT 10.0')
			I(1, 'Product', 'HFSS Version 2023.1.0')
		$end 'StartInfo'
		$begin 'TotalInfo'
			I(1, 'Elapsed Time', '02:05:28')
			I(1, 'ComEngine Memory', '404 M')
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
			ProfileItem('Machine', 0, 0, 0, 0, 0, 'I(5, 1, \'Name\', \'scurry\', 1, \'Memory\', \'23.6 GB\', 3, \'RAM Limit\', 90, \'%f%%\', 2, \'Cores\', 4, false, 1, \'Free Disk Space\', \'84.8 GB\')', false, true)
		$end 'ProfileGroup'
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Allow off core\', \'True\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Solution Basis Order\', \'1\')', false, true)
		ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(1, 0, \'Elapsed time : 00:00:02 , HFSS ComEngine Memory : 141 M\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Perform full validations with standard port validations\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2023
			MinorVer=1
			Name='Initial Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '07/21/2026 10:10:55')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:02:00')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			ProfileItem('Mesh', 2, 0, 1, 0, -1, 'I(2, 1, \'Type\', \'Link\', 1, \'Source\', \'This Project*, This Design* - Setup_10GHz : LastAdaptive\')', true, true)
			ProfileItem('Manual Refine', 35, 0, 34, 0, 901192, 'I(3, 2, \'Tetrahedra\', 908028, false, 2, \'Cores\', 1, false, 0, \'FeedSheetUniform_0p180mm, PortFeedUniform_0p180mm\')', true, true)
			ProfileItem('Simulation Setup', 15, 0, 15, 0, 1930436, 'I(1, 1, \'Disk\', \'0 Bytes\')', true, true)
			ProfileItem('Port Adapt', 35, 0, 34, 0, 1934856, 'I(2, 2, \'Tetrahedra\', 751953, false, 1, \'Disk\', \'9.35 MB\')', true, true)
			ProfileItem('Port Refine', 27, 0, 25, 0, 850972, 'I(2, 2, \'Tetrahedra\', 917546, false, 2, \'Cores\', 1, false)', true, true)
		$end 'ProfileGroup'
		$begin 'ProfileGroup'
			MajorVer=2023
			MinorVer=1
			Name='Adaptive Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '07/21/2026 10:12:56')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '02:03:25')
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
				ProfileItem('Domain Partitioning', 46, 0, 98, 0, 2487112, 'I(3, 2, \'Tetrahedra\', 760853, false, 2, \'Domain\', 3, false, 1, \'Disk\', \'40.3 MB\')', true, true)
				ProfileItem('Iterations', 7324, 0, 6762, 0, 2487112, 'I(5, 1, \'Total matrix size\', \'4505584\', 3, \'Average iterations/excitation\', 4.47656, \'%3.1f\', 2, \'Excitation\', 256, false, 2, \'Cores\', 1, false, 1, \'Disk\', \'30.8 MB\')', true, false)
				ProfileItem('Distributed Solve for - Adaptive_1 ', 7313, 0, 21318, 0, 0, 'I(1, 0, \'Maximum domain memory:  4.55 GB\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Average domain memory:  4.46 GB\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Total memory for all domains:  13.4 GB\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Total number of MPI processes:  4\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Total number of cores:  4\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2023
					MinorVer=1
					Name='Domain Group 1'
					$begin 'StartInfo'
						I(0, 'scurry')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, '')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('Matrix Assembly/Solve', 7313, 0, 7115, 0, 4769392, 'I(7, 1, \'Type\', \'MCS\', 2, \'Cores\', 1, false, 2, \'Tetrahedra\', 255369, false, 2, \'Lumped ports\', 83, false, 2, \'Matrix size\', 1499991, false, 3, \'Matrix bandwidth\', 20.1061, \'%5.1f\', 1, \'Disk\', \'0 Bytes\')', true, false)
					ProfileFootnote('I(1, 2, \'Num. Domain\', 1, false)', 0)
				$end 'ProfileGroup'
				$begin 'ProfileGroup'
					MajorVer=2023
					MinorVer=1
					Name='Domain Group 2'
					$begin 'StartInfo'
						I(0, 'scurry')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, '')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('Matrix Assembly/Solve', 7294, 0, 7098, 0, 4688432, 'I(7, 1, \'Type\', \'MCS\', 2, \'Cores\', 1, false, 2, \'Tetrahedra\', 253315, false, 2, \'Lumped ports\', 86, false, 2, \'Matrix size\', 1494054, false, 3, \'Matrix bandwidth\', 20.2007, \'%5.1f\', 1, \'Disk\', \'0 Bytes\')', true, false)
					ProfileFootnote('I(1, 2, \'Num. Domain\', 1, false)', 0)
				$end 'ProfileGroup'
				$begin 'ProfileGroup'
					MajorVer=2023
					MinorVer=1
					Name='Domain Group 3'
					$begin 'StartInfo'
						I(0, 'scurry')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, '')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('Matrix Assembly/Solve', 7303, 0, 7105, 0, 4562352, 'I(7, 1, \'Type\', \'MCS\', 2, \'Cores\', 1, false, 2, \'Tetrahedra\', 252169, false, 2, \'Lumped ports\', 87, false, 2, \'Matrix size\', 1489255, false, 3, \'Matrix bandwidth\', 20.2441, \'%5.1f\', 1, \'Disk\', \'0 Bytes\')', true, false)
					ProfileFootnote('I(1, 2, \'Num. Domain\', 1, false)', 0)
				$end 'ProfileGroup'
				ProfileItem('Data Transfer', 0, 0, 0, 0, 368280, 'I(1, 0, \'Adaptive Pass 1\')', true, true)
			$end 'ProfileGroup'
			ProfileFootnote('I(1, 0, \'Adaptive Passes did not converge\')', 1)
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
			ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:02\', 1, \'Total Memory\', \'141 MB\')', false, true)
			ProfileItem('Initial Meshing', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:02:00\', 1, \'Total Memory\', \'1.84 GB\')', false, true)
			ProfileItem('Adaptive Meshing', 0, 0, 0, 0, 0, 'I(5, 1, \'Elapsed Time\', \'02:03:25\', 1, \'Average memory/process\', \'3.94 GB\', 1, \'Max memory/process\', \'4.55 GB\', 2, \'Total number of processes\', 4, false, 2, \'Total number of cores\', 4, false)', false, true)
			ProfileFootnote('I(2, 2, \'Max solved tets\', 760853, false, 2, \'Max matrix size\', 4505584, false)', 0)
		$end 'ProfileGroup'
		ProfileFootnote('I(2, 1, \'Stop Time\', \'07/21/2026 12:16:21\', 1, \'Status\', \'Normal Completion\')', 0)
	$end 'ProfileGroup'
$end 'Profile'
