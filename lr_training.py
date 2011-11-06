from numpy import matrix
from sys import stdout, argv
import math
import json
import getopt

def sigmoid(z):
	z = z[0,0]
	a = 1.00/(1.00+math.e**-z)
	return a

def cost(x, y, t):
	m = len(x[:,1])
	a = 1.00/float(m)
	b = 0
	n = 0
	for i in range(m):
		n += 1
		b +=  -y[i]*math.log(sigmoid(t.T*x[i,:].T))-(1-y[i])*math.log(1-sigmoid(t.T*x[i,:].T))
	return -a*b

def gradient_calc(x, y, alpha, max_iter = 500):
	t0 = 0
	t1 = 0
	t2 = 0
	ts = matrix([t0, t1, t2]).T
	oj = cost(x,y, ts)[0,0]
	m = len(x[:,1])
	n = 0
	while 1:
		n += 1
		s = 0
		s1 = 0
		s2 = 0
		for i in range(m):
			s += sigmoid(ts.T*x[i,:].T)-y[i]
			s1 += (sigmoid(ts.T*x[i,:].T)-y[i])*x[i,1]
			s2 += (sigmoid(ts.T*x[i,:].T)-y[i])*x[i,2]


		s = s[0,0]
		s1 = s1[0,0]
		s2 = s2[0,0]
		t0 = t0 - alpha*1.00/m*s
		t1 = t1 - alpha*1.00/m*s1
		t2 = t2 - alpha*1.00/m*s2
		ts = matrix([t0, t1, t2]).T

		j = cost(x,y,ts)[0,0]

		if oj-j == 0:
			print "Naturally converged after %d iterations" % n
			return [t0, t1, t2]

		oj = j

		if n == max_iter and max_iter != 0:
			print "Reached iteration %d, stopping gradient descent" % n
			return [t0, t1, t2]

		stdout.write("Iteration: %d\r" % n)

alpha = 0.1
iters = 400
fn = None
opts, args = getopt.getopt(argv[1:], "a:i:f:", ["alpha=", "iters=", "file="])

for o,a in opts:
	if o in ("-a", "--alpha"):
		alpha = float(a)
	elif o in ("-i", "--iters"):
		iters = int(a)
	elif o in ("-f", "--file"):
		fn = a
	

if fn == None:
	print "Usage: python rl_training.py -f <file_where_to_save_thetas> [-i <max_iterations=400> -a <alpha=0.1>]"
	quit()

X = matrix("1 1 1; 1 1 0; 1 0 1; 1 0 0") # OR Matrix + 1s for Theta0
y = matrix("1; 1; 1; 0") # Solutions to OR logic operator
ts = gradient_calc(X, y, alpha, iters) # Gradient descent
fd = file(fn, "w")
fd.write(json.dumps(ts))
fd.close()
