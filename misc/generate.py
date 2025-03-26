#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor
from mpmath import mp

mp.dps = 30

precision = 17
zero_threshold = mp.mpf('1e-20')

nn = 101


def uniform(n):
	return [0] if n == 1 else [2 * mp.mpf(k) / (n-1) - 1 for k in range(n)]
def chebyshev(n):
	return [-mp.cos((2*k - 1) * mp.pi / (2*n)) for k in range(1, n + 1)]
def chebyshev_2(n):
	return [mp.sin(mp.pi / 2 * x) for x in uniform(n)]


def f2(x):
	return -3*mp.sin(10*x) + 10*mp.sin(mp.fabs(x) + x/2)


test_cases = []

for func in [f2]:
	for dist in [uniform]:
		for n in [9]:
			for a, b in [(-10, 10)]:
				test_cases.append((func, dist, n, a, b))

for func in [f2]:
	for dist in [chebyshev, chebyshev_2]:
		for n in [49, 99, 199]:
			for a, b in [(-10, 10)]:
				test_cases.append((func, dist, n, a, b))


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


def format_test_case(func, dist, X, Y, xx, yy):
	return '\n'.join([
		'[[test_cases]]',
		f'func = "{func.__name__}"',
		f'dist = "{dist.__name__}"',
		f'X = {format_array(X)}',
		f'Y = {format_array(Y)}',
		f'xx = {format_array(xx)}',
		f'yy = {format_array(yy)}',
	])


def barycentric(X, Y, xx, dist, epsilon):
	n = len(X)
	nn = len(xx)

	if dist == 'uniform':
		c = [(-1)**k * mp.binomial(n - 1, k) for k in range(n)]
	elif dist == 'chebyshev':
		c = [(-1)**k * mp.sin(((2 * k + 1) * mp.pi) / (2 * n)) for k in range(n)]
	elif dist == 'chebyshev_2':
		c = [(-1)**k * (mp.mpf('1/2') if k == 0 or k == n - 1 else mp.mpf(1)) for k in range(n)]
	else:
		raise ValueError(f'Unexpected distribution type: {dist!r}')

	numer = [mp.mpf(0)] * nn
	denom = [mp.mpf(0)] * nn
	exact = [-1] * nn

	for k in range(n):
		for i, xi in enumerate(xx):
			xdiff = xi - X[k]
			if mp.fabs(xdiff) < epsilon:
				exact[i] = k
			else:
				temp = c[k] / xdiff
				numer[i] += temp * Y[k]
				denom[i] += temp

	yy = [Y[exact[i]] if exact[i] != -1 else numer[i] / denom[i] for i in range(nn)]
	return yy


def generate_test_case(params):
	func, dist, n, a, b = params
	X = stretched(dist(n), a, b)
	Y = [func(x) for x in X]
	xx = stretched(uniform(nn), a, b)
	yy = barycentric(X, Y, xx, dist.__name__, zero_threshold)
	return format_test_case(func, dist, X, Y, xx, yy)


def generate_test_cases():
	with ProcessPoolExecutor() as executor:
		results = list(executor.map(generate_test_case, test_cases))
	return '\n\n'.join(results)


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