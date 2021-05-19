from pandas import DataFrame
import sys
import csv

def prRed(skk): return "\033[91m{}\033[00m".format(skk)
def prGreen(skk): return "\033[92m{}\033[00m".format(skk)
def prYellow(skk): return "\033[33m{}\033[0m".format(skk)
def prLightPurple(skk): return "\033[94m{}\033[00m".format(skk)
def prPurple(skk): return "\033[95m{}\033[00m".format(skk)
def prCyan(skk): return "\033[96m{}\033[00m".format(skk)
def prLightGray(skk): return "\033[97m{}\033[00m".format(skk)
def prBlack(skk): return "\033[98m {}\033[00m".format(skk)

def _mean(arr):
    total = 0
    for i in arr:
        total += i
    return total / len(arr)


def normalizeElem(list, elem):
    return ((elem - min(list)) / (max(list) - min(list)))


def denormalizeElem(list, elem):
    return ((elem * (max(list) - min(list))) + min(list))


def error_exit(string):
    """
    Print + quit
    """
    print(string)
    sys.exit(0)


def normalisation(s):
    return [((_ - min(s)) / (max(s) - min(s))) for _ in s]


def raw_estimated_price(t0, x, t1):
    return t0 + x * t1


def estimated_price(t0, t1, x, X, Y):
    # print(f'({x} - {min(X)}) / ({max(X)} - {min(X)}) == { (x - min(X)) / (max(X) - min(X))}')
    price_ranged = raw_estimated_price(t0, (x - min(X)) / (max(X) - min(X)), t1)
    # print(price_ranged)
    # print(f"(max(Y) - min(Y)) == {(max(Y) - min(Y))}")
    return price_ranged * (max(Y) - min(Y)) + min(Y)


def cost(X, Y, th):
    """
    MSE
    """
    dfX = DataFrame(X, columns=['X'])
    dfY = DataFrame(Y, columns=['Y'])
    return ((th[1] * dfX['X'] + th[0] - dfY['Y']) ** 2).sum() / len(dfX['X'])


def getData(file):
    mileages = []
    prices = []
    with open(file, 'r') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for row in csvReader:
            mileages.append(row[0])
            prices.append(row[1])

    mileages.pop(0)
    prices.pop(0)
    for i in range(len(mileages)):
        mileages[i] = eval(mileages[i])
        prices[i] = eval(prices[i])
    return (mileages, prices)


def get_gradient_csv(input):
    try:
        thetas = {}
        file = open(input, 'r')
        lines = file.readlines()
        for line in lines:
            thetas[line.strip().split(':')[0]] = float(line.strip().split(':')[1])
    # print(thetas)
    except:
        thetas = {'T0': 0, 'T1': 0}
    return thetas


def set_gradient_csv(output, t0, t1):
    # Theta file
    try:
        with open(output, "w+") as f:
            f.write('T0:{}\nT1:{}\n'.format(t0, t1))
    except:
        error_exit('Wrong file')


def debug(message):
    """
        Verbosity for debug msg
    """
    print("\033[33m{:s}\033[0m".format(message))


def normal(message):
    """
        Verbosity for normal msg
    """
    print(message)


def success(message):
    """
        Verbosity for success msg
    """
    print("\033[32m{:s}\033[0m".format(message))


def verbose(message):
    """
        Verbosity for info msg
    """
    print("\033[38;5;247m{:s}\033[0m".format(message))


def error(message):
    """
        Verbosity for error msg
    """
    print("\033[31m{:s}\033[0m".format(message))
