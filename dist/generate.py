#!/usr/bin/env python3
import argparse
import inspect
from mpmath import mp

mp.dps = 20

precision = 17
zero_threshold = mp.mpf('1e-18')

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


def quadratic(n):
	return [(x + 1)**2 - 1 if x <= 0 else -(x - 1)**2 + 1 for x in uniform(n)]


def cubic(n):
	return [-0.5 * x**3 + 1.5 * x for x in uniform(n)]


def chebyshev(n):
	return [-mp.cos((2*k - 1) * mp.pi / (2*n)) for k in range(1, n + 1)]


def chebyshev_stretched(n):
	return stretched(chebyshev(n))


def chebyshev_augmented(n):
	return [] if n == 0 else [0] if n == 1 else [-1] + chebyshev(n - 2) + [1]


def chebyshev_2(n):
	return [mp.sin(mp.pi / 2 * x) for x in uniform(n)]


def chebyshev_3(n):
	return [mp.cos(((2*n - 1 - 2*k) * mp.pi) / (2*n - 1)) for k in range(n)]


def chebyshev_3_stretched(n):
	return stretched(chebyshev_3(n))


def chebyshev_4(n):
	return [mp.cos(((2*n - 2 - 2*k) * mp.pi) / (2*n - 1)) for k in range(n)]


def chebyshev_4_stretched(n):
	return stretched(chebyshev_4(n))


def chebyshev_ellipse(n, ratio):
	return [mp.sign(2*k+1 - n) / mp.sqrt(1 + (mp.tan(mp.pi * (2*mp.mpf(k) + 1) / (2*n)) / ratio) ** 2) for k in range(n)]


def chebyshev_ellipse_stretched(n, ratio):
	return stretched(chebyshev_ellipse(n, ratio))


def chebyshev_ellipse_augmented(n, ratio):
	return [] if n == 0 else [0] if n == 1 else [-1] + chebyshev_ellipse(n - 2, ratio) + [1]


def chebyshev_ellipse_2(n, ratio):
	return [0] if n == 1 else [mp.sign(2*k+1 - n) / mp.sqrt(1 + (mp.tan(mp.pi * mp.mpf(k) / (n-1)) / ratio) ** 2) for k in range(n)]


def chebyshev_ellipse_3(n, ratio):
	return [(-1 if theta < mp.pi/2 else 1) / mp.sqrt(1 + (mp.tan(theta) / ratio) ** 2) for theta in (mp.pi * (2*k) / (2*n - 1) for k in range(n))]


def chebyshev_ellipse_3_stretched(n, ratio):
	return stretched(chebyshev_ellipse_3(n, ratio))


def chebyshev_ellipse_4(n, ratio):
	return [(-1 if theta < mp.pi/2 else 1) / mp.sqrt(1 + (mp.tan(theta) / ratio) ** 2) for theta in (mp.pi * (2*k + 1) / (2*n - 1) for k in range(n))]


def chebyshev_ellipse_4_stretched(n, ratio):
	return stretched(chebyshev_ellipse_4(n, ratio))


def logistic(n, steepness):
	return [2 / (1 + mp.exp(-steepness * x)) - 1 for x in uniform(n)]


def logistic_stretched(n, steepness):
	return stretched(logistic(n, steepness))


def erf(n, steepness):
	return [mp.erf(steepness * x) for x in uniform(n)]


def erf_stretched(n, steepness):
	return stretched(erf(n, steepness))


def generate_test_cases():
	mapping_intervals_section = 'mapping_intervals = [\n' + '\n'.join([f'\t[{i[0]}, {i[1]}],' for i in mapping_intervals]) + '\n]'

	functions = [uniform, quadratic, cubic, chebyshev, chebyshev_stretched, chebyshev_augmented, chebyshev_2,
				chebyshev_3, chebyshev_3_stretched, chebyshev_4, chebyshev_4_stretched,
				chebyshev_ellipse, chebyshev_ellipse_stretched, chebyshev_ellipse_augmented, chebyshev_ellipse_2,
				chebyshev_ellipse_3, chebyshev_ellipse_3_stretched, chebyshev_ellipse_4, chebyshev_ellipse_4_stretched,
				logistic, logistic_stretched, erf, erf_stretched]
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