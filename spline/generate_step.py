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
types = ['left', 'middle', 'right']
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


def format_test_case(func, dist, type, X, Y, xx, yy):
	return '\n'.join([
		'[[test_cases]]',
		f'func = "{func.__name__}"',
		f'dist = "{dist.__name__}"',
		f'type = "{type}"',
		f'X = {format_array(X)}',
		f'Y = {format_array(Y)}',
		f'xx = {format_array(xx)}',
		f'yy = {format_array(yy)}',
	])


def generate_test_case(params):
	func, dist, type, n, a, b = params
	X = stretched(dist(n), a, b)
	Y = [func(x) for x in X]
	xx = stretched(uniform(nn), a, b)
	yy = []
	cur_segment = 0
	for x in xx:
		while cur_segment + 1 < len(X) - 1 and X[cur_segment + 1] <= x:
			cur_segment += 1
		if mp.fabs(x - X[cur_segment]) < float64_eps:
			yy.append(Y[cur_segment])
			continue
		if mp.fabs(x - X[cur_segment+1]) < float64_eps:
			yy.append(Y[cur_segment+1])
			continue
		match type:
			case 'left':
				yy.append(Y[cur_segment])
			case 'middle':
				yy.append((Y[cur_segment] + Y[cur_segment+1]) / 2)
			case 'right':
				yy.append(Y[cur_segment+1])
			case _:
				raise ValueError(f'Unexpected type: {type!r}')
	return format_test_case(func, dist, type, X, Y, xx, yy)


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