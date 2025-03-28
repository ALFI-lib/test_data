#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor
from mpmath import mp
import sys

mp.dps = 20

precision = 17
zero_threshold = mp.mpf('1e-18')
float64_eps = mp.mpf(sys.float_info.epsilon)

nn = 23

def exp(x):
	return mp.exp(x)
def sin(x):
	return mp.sin(x)
def cos(x):
	return mp.cos(x)
def f1(x):
	return mp.fabs(x) + x/2 - x*x
def f2(x):
	return -3*mp.sin(10*x) + 10*mp.sin(mp.fabs(x) + x/2)

def uniform(n):
	return [mp.mpf(0)] if n == 1 else [2 * mp.mpf(k) / (n-1) - 1 for k in range(n)]
def chebyshev(n):
	return [-mp.cos((2*k - 1) * mp.pi / (2*n)) for k in range(1, n + 1)]
def chebyshev_2(n):
	return [mp.sin(mp.pi / 2 * x) for x in uniform(n)]

test_cases = []

functions = [f2]
distributions = [uniform, chebyshev, chebyshev_2]
types = ['semi-not-a-knot', 'semi-natural']
point_counts = [11]
intervals = [(-10, 10)]

for func in functions:
	for dist in distributions:
		for type in types:
			for n in point_counts:
				for a, b in intervals:
					test_cases.append((func, dist, type, n, a, b))


def stretched(points, a, b):
	if not points:
		return []
	if len(points) == 1 or min(points) == max(points):
		return [(a+b)/2] * len(points)
	return [a + (p-min(points)) * (b-a) / (max(points)-min(points)) for p in points]


def format_number(number):
	return '0' if abs(number) < zero_threshold else mp.nstr(number, n=precision).removesuffix('.0')


def format_array(array):
	return '[' + ', '.join(format_number(x) for x in array) + ']'


def format_test_case(func, dist, type, X, Y, coeffs, xx, yy):
	return '\n'.join([
		'[[test_cases]]',
		f'func = "{func.__name__}"',
		f'dist = "{dist.__name__}"',
		f'type = "{type}"',
		f'X = {format_array(X)}',
		f'Y = {format_array(Y)}',
		f'coeffs = {format_array(coeffs)}',
		f'xx = {format_array(xx)}',
		f'yy = {format_array(yy)}',
	])


def generate_test_case(params):
	func, dist, type, n, a, b = params
	X = stretched(dist(n), a, b)
	Y = [func(x) for x in X]
	dX = [X[i+1] - X[i] for i in range(n - 1)]
	dY = [Y[i+1] - Y[i] for i in range(n - 1)]
	if type == 'semi-not-a-knot':
		c = (dY[1]/dX[1] - dY[0]/dX[0]) / (dX[0] + dX[1])
		not_a_knot_start = [[c, dY[0]/dX[0] - c*dX[0], Y[0]]] + [[0, 0, 0] for _ in range(n - 2)]
		c = (dY[n-2]/dX[n-2] - dY[n-3]/dX[n-3]) / (dX[n-3] + dX[n-2])
		not_a_knot_end = [[0, 0, 0] for _ in range(n - 2)] + [[c, dY[n-2]/dX[n-2] - c*dX[n-2], Y[n-2]]]
		spline1 = not_a_knot_start
		spline2 = not_a_knot_end
	elif type == 'semi-natural':
		natural_start = [[0, dY[0]/dX[0], Y[0]]] + [[0, 0, 0] for _ in range(n - 2)]
		natural_end = [[0, 0, 0] for _ in range(n - 2)] + [[0, dY[n-2]/dX[n-2], Y[n-2]]]
		spline1 = natural_start
		spline2 = natural_end
	else:
		raise ValueError(f'Unexpected type: {type!r}')
	for i in range(1, n - 1):
		spline1[i][2] = Y[i]
		spline1[i][1] = 2*dY[i-1]/dX[i-1] - spline1[i-1][1]
		spline1[i][0] = (dY[i]/dX[i] - spline1[i][1]) / dX[i]
		j = n - 2 - i
		spline2[j][2] = Y[j]
		spline2[j][1] = 2*dY[j]/dX[j] - spline2[j+1][1]
		spline2[j][0] = (spline2[j+1][1] - dY[j]/dX[j]) / dX[j]
	coeffs = [c for s1, s2 in zip(spline1, spline2) for c in [(c1+c2)/2 for c1, c2 in zip(s1, s2)]]
	xx = stretched(uniform(nn), a, b)
	yy = []
	cur_segment = 0
	for x in xx:
		while cur_segment + 1 < len(X) - 1 and X[cur_segment + 1] <= x:
			cur_segment += 1
		x_seg = x - X[cur_segment]
		yy.append((coeffs[3*cur_segment+0] * x_seg + coeffs[3*cur_segment+1]) * x_seg + coeffs[3*cur_segment+2])
	return format_test_case(func, dist, type, X, Y, coeffs, xx, yy)


def generate_test_cases():
	with ProcessPoolExecutor() as executor:
		return '\n\n'.join(executor.map(generate_test_case, test_cases))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--output', type=str, help='Output file')
	args = parser.parse_args()
	output = generate_test_cases()
	if not args.output:
		print(output, end='')
	else:
		with open(args.output, 'w') as file:
			file.write(output)


if __name__ == '__main__':
	main()