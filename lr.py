from numpy import matrix
import numpy
import math
from sys import argv
import re
import json

def sigmoid(z):
	z = z[0,0]
	a = 1.00/(1.00+math.e**-z)
	return a


def h(x, t): #hypothesis
	sm = sigmoid(t.T*x.T)
	if sm >= 0.5:
		return 1
	else:
		return 0


#ts = matrix("-37.237; 24.871; 24.871") <-- Best thetas
if len(argv) < 3:
	print "Usage: rl.py <filename_where_to_get_thetas> <(0|1):(0|1)>"
	quit()

fd = file(argv[1], "r")
ts =  fd.read()
fd.close()
ts = matrix(json.loads(ts))
print ts

ds =  re.findall("([0-1]):([0-1])", argv[2])
if ds == []:
	print "Error: only '0' or '1' digits"
	quit()
	
ds = ds[0]
f = [1, int(ds[0]), int(ds[1])] # Add Theta0 term
print matrix(f)
print "%d OR %d = %d" % (int(ds[0]), int(ds[1]), h(matrix(f), ts.T))
