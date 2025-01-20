#!/usr/bin/env python3
import argparse
import inspect
from mpmath import mp

mp.dps = 100

precision = 17
zero_threshold = mp.mpf('1e-80')

a, b = -1, 1

max_n = 6

ratios = [mp.mpf('0.1'), mp.mpf('0.25'), mp.mpf('0.5'), mp.mpf('1'), mp.mpf('2'), mp.mpf('4'), mp.mpf('10')]
steepnesses = [mp.mpf('0.1'), mp.mpf('0.25'), mp.mpf('0.5'), mp.mpf('1'), mp.mpf('2'), mp.mpf('4'), mp.mpf('10')]
mapping_intervals = [
	[0, 1],
	[-2, 2],
	[-1000, -900],
	[1000, 2000],
	[-9999, 9999],
]


def format_number(number):
	return '0' if abs(number) < zero_threshold else mp.nstr(number, n=precision).removesuffix('.0')


def format_test_case(n, a, b, points, **kwargs):
	params = ''.join([f', {key}={format_number(value)}' for key, value in kwargs.items() if value is not None])
	points_str = ', '.join(format_number(x) for x in points)
	return f'\t{{ n = {n}, a = {format_number(a)}, b = {format_number(b)}{params}, expected = [{points_str}] }},'


def stretched(points):
	if not points:
		return []
	if len(points) == 1 or min(points) == max(points):
		return [(a+b)/2] * len(points)
	return [a + (p-min(points)) * (b-a) / (max(points)-min(points)) for p in points]


def uniform(n):
	return [0] if n == 1 else [2 * mp.mpf(k) / (n-1) - 1 for k in range(n)]


def chebyshev(n):
	return [-mp.cos((2*k - 1) * mp.pi / (2*n)) for k in range(1, n + 1)]


def chebyshev_stretched(n):
	return stretched(chebyshev(n))


def chebyshev_ellipse(n, ratio):
	return [mp.sign(2*k+1 - n) / mp.sqrt(1 + (mp.tan(mp.pi * (2*mp.mpf(k) + 1) / (2*n)) / ratio) ** 2) for k in range(n)]


def chebyshev_ellipse_stretched(n, ratio):
	return stretched(chebyshev_ellipse(n, ratio))


def circle_proj(n):
	return [0] if n == 1 else [mp.sin(mp.pi / 2 * (2 * mp.mpf(k) / (n-1) - 1)) for k in range(n)]


def ellipse_proj(n, ratio):
	return [0] if n == 1 else [mp.sign(2*k+1 - n) / mp.sqrt(1 + (mp.tan(mp.pi * mp.mpf(k) / (n-1)) / ratio) ** 2) for k in range(n)]


def sigmoid(n, steepness):
	return [0] if n == 1 else [2 / (1 + mp.exp(-steepness * (2 * mp.mpf(k) / (n-1) - 1))) - 1 for k in range(n)]


def sigmoid_stretched(n, steepness):
	return stretched(sigmoid(n, steepness))


def erf(n, steepness):
	return [0] if n == 1 else [mp.erf(steepness * (2 * mp.mpf(k) / (n-1) - 1)) for k in range(n)]


def erf_stretched(n, steepness):
	return stretched(erf(n, steepness))


def generate_test_cases():
	mapping_intervals_section = 'mapping_intervals = [\n' + '\n'.join([f'\t[{i[0]}, {i[1]}],' for i in mapping_intervals]) + '\n]'

	functions = [uniform, chebyshev, chebyshev_stretched, chebyshev_ellipse, chebyshev_ellipse_stretched,
				circle_proj, ellipse_proj, sigmoid, sigmoid_stretched, erf, erf_stretched]
	sections = []

	for func in functions:
		func_cases = []
		for n in range(max_n + 1):
			func_parameter_names = [p.name for p in inspect.signature(func).parameters.values()]
			if 'ratio' in func_parameter_names:
				for ratio in ratios:
					func_cases.append(format_test_case(n, a, b, func(n, ratio), ratio=ratio))
			elif 'steepness' in func_parameter_names:
				for steepness in steepnesses:
					func_cases.append(format_test_case(n, a, b, func(n, steepness), steepness=steepness))
			else:
				func_cases.append(format_test_case(n, a, b, func(n)))
		section = f'[{func.__name__}]\ntest_cases = [\n' + '\n'.join(func_cases) + '\n]'
		sections.append(section)

	return mapping_intervals_section + '\n\n' + '\n\n'.join(sections)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--output', type=str, help="Output file")
	args = parser.parse_args()
	if not args.output:
		print(generate_test_cases(), end='')
	else:
		with open(args.output, 'w') as file:
			file.write(generate_test_cases())


if __name__ == '__main__':
	main()