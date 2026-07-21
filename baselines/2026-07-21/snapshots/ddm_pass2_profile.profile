$begin 'Profile'
	$begin 'ProfileGroup'
		MajorVer=2023
		MinorVer=1
		Name='Solution Process'
		$begin 'StartInfo'
			I(1, 'Start Time', '07/21/2026 12:21:55')
			I(1, 'Host', 'SCURRY')
			I(1, 'Processor', '12')
			I(1, 'OS', 'NT 10.0')
			I(1, 'Product', 'HFSS Version 2023.1.0')
		$end 'StartInfo'
		$begin 'TotalInfo'
			I(1, 'Elapsed Time', '02:44:54')
			I(1, 'ComEngine Memory', '379 M')
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
			ProfileItem('Machine', 0, 0, 0, 0, 0, 'I(5, 1, \'Name\', \'scurry\', 1, \'Memory\', \'23.6 GB\', 3, \'RAM Limit\', 90, \'%f%%\', 2, \'Cores\', 4, false, 1, \'Free Disk Space\', \'84.7 GB\')', false, true)
		$end 'ProfileGroup'
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Allow off core\', \'True\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Solution Basis Order\', \'1\')', false, true)
		ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(1, 0, \'Elapsed time : 00:00:05 , HFSS ComEngine Memory : 141 M\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Perform full validations with standard port validations\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2023
			MinorVer=1
			Name='Initial Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '07/21/2026 12:22:00')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:03:56')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			ProfileItem('Mesh', 1, 0, 1, 0, -1, 'I(2, 1, \'Type\', \'Link\', 1, \'Source\', \'This Project*, This Design* - Setup_10GHz : LastAdaptive\')', true, true)
			ProfileItem('Manual Refine', 66, 0, 62, 0, 878004, 'I(3, 2, \'Tetrahedra\', 908028, false, 2, \'Cores\', 1, false, 0, \'FeedSheetUniform_0p180mm, PortFeedUniform_0p180mm\')', true, true)
			ProfileItem('Lambda Refine', 23, 0, 22, 0, 1450596, 'I(2, 2, \'Tetrahedra\', 909820, false, 2, \'Cores\', 2, false)', true, true)
			ProfileItem('Simulation Setup', 25, 0, 24, 0, 1934320, 'I(1, 1, \'Disk\', \'0 Bytes\')', true, true)
			ProfileItem('Port Adapt', 80, 0, 76, 0, 1939156, 'I(2, 2, \'Tetrahedra\', 753745, false, 1, \'Disk\', \'9.35 MB\')', true, true)
			ProfileItem('Port Refine', 32, 0, 31, 0, 852976, 'I(2, 2, \'Tetrahedra\', 919339, false, 2, \'Cores\', 1, false)', true, true)
		$end 'ProfileGroup'
		$begin 'ProfileGroup'
			MajorVer=2023
			MinorVer=1
			Name='Adaptive Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '07/21/2026 12:25:56')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '02:40:52')
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
				ProfileItem('Domain Partitioning', 47, 0, 99, 0, 2505540, 'I(3, 2, \'Tetrahedra\', 762641, false, 2, \'Domain\', 3, false, 1, \'Disk\', \'41.4 MB\')', true, true)
				ProfileItem('Iterations', 4519, 0, 4154, 0, 2505540, 'I(5, 1, \'Total matrix size\', \'4519800\', 3, \'Average iterations/excitation\', 5.22656, \'%3.1f\', 2, \'Excitation\', 256, false, 2, \'Cores\', 1, false, 1, \'Disk\', \'32.9 MB\')', true, false)
				ProfileItem('Distributed Solve for - Adaptive_1 ', 4515, 0, 13508, 0, 0, 'I(1, 0, \'Maximum domain memory:  5.32 GB\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Average domain memory:  5 GB\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Total memory for all domains:  15 GB\')', false, true)
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
					ProfileItem('Matrix Assembly/Solve', 4513, 0, 4503, 0, 4965384, 'I(7, 1, \'Type\', \'MCS\', 2, \'Cores\', 1, false, 2, \'Tetrahedra\', 254309, false, 2, \'Lumped ports\', 91, false, 2, \'Matrix size\', 1504955, false, 3, \'Matrix bandwidth\', 20.2911, \'%5.1f\', 1, \'Disk\', \'0 Bytes\')', true, false)
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
					ProfileItem('Matrix Assembly/Solve', 4515, 0, 4505, 0, 5581372, 'I(7, 1, \'Type\', \'MCS\', 2, \'Cores\', 1, false, 2, \'Tetrahedra\', 254237, false, 2, \'Lumped ports\', 87, false, 2, \'Matrix size\', 1501085, false, 3, \'Matrix bandwidth\', 20.2332, \'%5.1f\', 1, \'Disk\', \'0 Bytes\')', true, false)
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
					ProfileItem('Matrix Assembly/Solve', 4512, 0, 4500, 0, 5166672, 'I(7, 1, \'Type\', \'MCS\', 2, \'Cores\', 1, false, 2, \'Tetrahedra\', 254095, false, 2, \'Lumped ports\', 78, false, 2, \'Matrix size\', 1489340, false, 3, \'Matrix bandwidth\', 20.0331, \'%5.1f\', 1, \'Disk\', \'0 Bytes\')', true, false)
					ProfileFootnote('I(1, 2, \'Num. Domain\', 1, false)', 0)
				$end 'ProfileGroup'
				ProfileItem('Data Transfer', 0, 0, 0, 0, 27396, 'I(1, 0, \'Adaptive Pass 1\')', true, true)
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
				ProfileItem('Adaptive Refine', 18, 0, 18, 0, 859664, 'I(2, 2, \'Tetrahedra\', 957472, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem('Domain Partitioning', 27, 0, 60, 0, 2605616, 'I(3, 2, \'Tetrahedra\', 794669, false, 2, \'Domain\', 3, false, 1, \'Disk\', \'43.7 MB\')', true, true)
				ProfileItem('Iterations', 5003, 0, 4550, 0, 2605616, 'I(5, 1, \'Total matrix size\', \'4710680\', 3, \'Average iterations/excitation\', 4.91406, \'%3.1f\', 2, \'Excitation\', 256, false, 2, \'Cores\', 1, false, 1, \'Disk\', \'32.4 MB\')', true, false)
				ProfileItem('Distributed Solve for - Adaptive_2 ', 4994, 0, 14790, 0, 0, 'I(1, 0, \'Maximum domain memory:  5.34 GB\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Average domain memory:  4.98 GB\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Total memory for all domains:  14.9 GB\')', false, true)
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
					ProfileItem('Matrix Assembly/Solve', 4984, 0, 4931, 0, 4886724, 'I(7, 1, \'Type\', \'MCS\', 2, \'Cores\', 1, false, 2, \'Tetrahedra\', 265254, false, 2, \'Lumped ports\', 88, false, 2, \'Matrix size\', 1568650, false, 3, \'Matrix bandwidth\', 20.2732, \'%5.1f\', 1, \'Disk\', \'0 Bytes\')', true, false)
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
					ProfileItem('Matrix Assembly/Solve', 4974, 0, 4918, 0, 5596428, 'I(7, 1, \'Type\', \'MCS\', 2, \'Cores\', 1, false, 2, \'Tetrahedra\', 264712, false, 2, \'Lumped ports\', 90, false, 2, \'Matrix size\', 1567096, false, 3, \'Matrix bandwidth\', 20.308, \'%5.1f\', 1, \'Disk\', \'0 Bytes\')', true, false)
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
					ProfileItem('Matrix Assembly/Solve', 4994, 0, 4941, 0, 5170180, 'I(7, 1, \'Type\', \'MCS\', 2, \'Cores\', 1, false, 2, \'Tetrahedra\', 264703, false, 2, \'Lumped ports\', 78, false, 2, \'Matrix size\', 1550142, false, 3, \'Matrix bandwidth\', 20.0037, \'%5.1f\', 1, \'Disk\', \'0 Bytes\')', true, false)
					ProfileFootnote('I(1, 2, \'Num. Domain\', 1, false)', 0)
				$end 'ProfileGroup'
				ProfileItem('Data Transfer', 0, 0, 0, 0, 351208, 'I(1, 0, \'Adaptive Pass 2\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 3, \'Max Mag. Delta S\', 0.29495, \'%.5f\')', false, true)
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
			ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:05\', 1, \'Total Memory\', \'141 MB\')', false, true)
			ProfileItem('Initial Meshing', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:03:56\', 1, \'Total Memory\', \'1.85 GB\')', false, true)
			ProfileItem('Adaptive Meshing', 0, 0, 0, 0, 0, 'I(5, 1, \'Elapsed Time\', \'02:40:52\', 1, \'Average memory/process\', \'4.35 GB\', 1, \'Max memory/process\', \'5.34 GB\', 2, \'Total number of processes\', 4, false, 2, \'Total number of cores\', 4, false)', false, true)
			ProfileFootnote('I(2, 2, \'Max solved tets\', 794669, false, 2, \'Max matrix size\', 4710680, false)', 0)
		$end 'ProfileGroup'
		ProfileFootnote('I(2, 1, \'Stop Time\', \'07/21/2026 15:06:49\', 1, \'Status\', \'Normal Completion\')', 0)
	$end 'ProfileGroup'
$end 'Profile'
