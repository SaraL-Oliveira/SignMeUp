BEGIN {
	print "Time | Acc_X | Acc_Y | Acc_Z | Ang_X | Ang_Y | Ang_Z"
	
}

{	
	
	if (match($0, /Left,([0-9\-]+),([0-9\-]+),([0-9\-]+),([0-9\-]+),([0-9\-]+),([0-9\-]+),([0-9\-]+)/, arr)){

		
		print arr[1]"|"arr[2]"|"arr[3]"|"arr[4]"|"arr[5]"|"arr[6]"|"arr[7]
		
		next
	}
}

END{
}