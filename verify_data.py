#!/usr/bin/env python3
import subprocess
import tempfile

def execute_command(command):
	print(' '.join(command), flush=True)
	subprocess.check_call(command)

def check_data(original, generator):
	with tempfile.NamedTemporaryFile() as tmpfile:
		execute_command([generator, '-o', tmpfile.name])
		execute_command(['diff', original, tmpfile.name])

check_data('dist/dist.toml', 'dist/generate.py')
check_data('poly/poly.toml', 'poly/generate.py')
check_data('misc/barycentric.toml', 'misc/generate.py')
check_data('spline/step.toml', 'spline/generate_step.py')
check_data('spline/linear.toml', 'spline/generate_linear.py')
check_data('spline/quadratic.toml', 'spline/generate_quadratic.py')
check_data('spline/cubic.toml', 'spline/generate_cubic.py')