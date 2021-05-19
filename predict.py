import matplotlib.pyplot as plt
import pandas as pd

import argparse
import numpy as np
import sys


def error_exit(string):
	"""
	Print + quit
	"""
	print(string)
	sys.exit(0)


def file_check(file):
	"""
	Checking if file is correct
	"""
	try:
		with open(file, 'r') as f:
			fl = f.readlines()
			for l in fl[1:]:
				L = l[:-1].split(',')
				if not L[0].isnumeric() or not L[1].isnumeric():
					return 0
			return 1
	except:
		error_exit('No data')


def check_number(f, n):
	"""
	Check if number is correct
	"""
	try:
		n = f(n)
	except:
		error_exit("Value error")
	return n


if __name__ == '__main__':
	# Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('mileage', type=int, help='mileage to predict')
	parser.add_argument('file', type=str, help='text file for input', default=None)
	parser.add_argument('-sc', '--scatter', help='data scatter plot', type=str)
	args = parser.parse_args()
	# Check if theta is here
	try:
		with open(args.file, 'r') as f:
			if f.mode == 'r':
				theta = []
				fl = f.readlines()
				for x in fl:
					theta.append(check_number(float, x.split(':')[1]))
	except:
		theta = [0, 0]
	print('Theta0: ', theta[0])
	print('Theta1: ', theta[1])

	# Mileage price calculation
	d = args.mileage
	d = check_number(float, d)
	p = theta[0] + theta[1] * d
	print('The price of this care is :', p, 'euros')

	# Check if scatter argument is here
	if args.scatter:
		if not file_check(args.scatter):
			error_exit('Bad File Format')
		df = pd.read_csv(args.scatter)
		km = df.columns[0]
		price = df.columns[1]
		plt.scatter(df[km], df[price])
		X, Y = df[km], df[price]
	else:
		X = np.linspace(0, 250000, num=25)
		price = None
		km = None

	# Plotting
	plt.plot(d, p, '*', markersize=12, color='red')
	if d > max(X):
		plt.plot(
			pd.DataFrame([i for i in range(int(d) + 10000)], columns=['KM']),
			theta[0] + theta[1] * pd.DataFrame([i for i in range(int(d) + 10000)], columns=['KM']),
			color='green'
		)
	else:
		plt.plot(X, theta[0] + theta[1] * X, color='green')
	if price and km:
		plt.ylabel(price)
		plt.xlabel(km)
		plt.title(price + ' = f(' + km + ')')
	plt.savefig('PredictGraph.png')
