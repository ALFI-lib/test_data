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