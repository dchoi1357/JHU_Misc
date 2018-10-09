import time

for x in range(20):
	time.sleep(3)
	print('%10d document processed'%x, end='\r')
	print('\b' * (10+len(' document processed')), end='\r' ),
