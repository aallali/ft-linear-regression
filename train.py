from util import *
from reprint import output
import matplotlib.pyplot as plt
import argparse
import imageio
import signal
import os


class ft_linear_regression:
    def __init__(self, rate, iter, input, output, po, pn, history, live):

        # Declare the figure where the data will be plotted
        self.figure, self.axis = plt.subplots(2, 2, figsize=(10, 10))

        # Import data sets and split its columns to X,Y normalized, and x,y original
        self.data = getData(input)
        self.x = self.data[0]
        self.y = self.data[1]
        self.X = normalisation(self.x)
        self.Y = normalisation(self.y)

        # Theta normalized / _Theta denormalized
        self._T0 = 0
        self._T1 = 0
        self.T0 = 1
        self.T1 = 1

        # M: Length of datasets | C: history of cost | images: all iteration shots |
        #                       | MSE: Mean Square Error Percentage | RMSE: MSE**2 Percentage
        self.M = len(self.x)
        self.C = []
        self.images = []
        self.RMSE = None
        self.MSE = None

        # delta MSE to calculate the progress of accurracy
        self.prev_mse = 0.0
        self.cur_mse = self.cost()
        self.delta_mse = self.cur_mse

        # learning_rate: or alpha | iterations: number of loop needed to train that fking model
        #                         | output: output file name to store final Theta0/Theta1
        self.learning_rate = rate
        self.iterations = 0
        self.max_iterations = iter
        self.output = output

        # visualisation
        #   po: plot original data
        #   pn: plot normalized data
        #   history: plot history of cost
        #   live: live watch of the training model
        self.po = po
        self.pn = pn
        self.history = history
        self.live = live

    def RMSE_percent(self):
        self.RMSE = 100 * (1 - self.cost() ** 0.5)
        return self.RMSE

    def MSE_percent(self):
        self.MSE = 100 * (1 - self.cost())
        return self.MSE

    def cost(self):
        """
        MSE
        """
        dfX = DataFrame(self.X, columns=['X'])
        dfY = DataFrame(self.Y, columns=['Y'])
        return ((self.T1 * dfX['X'] + self.T0 - dfY['Y']) ** 2).sum() / self.M

    def estimatePrice(self, t0, t1, mileage):
        return ((t0 + (t1 * float(mileage))))

    def live_update(self, output_lines):
        deltaX = max(self.x) - min(self.x)
        deltaY = max(self.y) - min(self.y)
        self._T1 = deltaY * self.T1 / deltaX
        self._T0 = ((deltaY * self.T0) + min(self.y) - self.T1 * (deltaY / deltaX) * min(self.x))
        output_lines[prCyan('    Theta0           ')] = str(self.T0)
        output_lines[prCyan('    Theta1           ')] = str(self.T1)
        output_lines[prCyan('    RMSE             ')] = f'{round(self.RMSE_percent(), 2)} %'
        output_lines[prCyan('    MSE              ')] = f'{round(self.MSE_percent(), 2)} %'
        output_lines[prCyan('    Delta MSE        ')] = str(self.delta_mse)
        output_lines[prCyan('    Iterations       ')] = str(self.iterations)

    def condition_to_stop_training(self):
        if self.max_iterations == 0:
            return self.delta_mse > 0.0000001 or self.delta_mse < -0.0000001
        else:
            return self.iterations < self.max_iterations

    def gradient_descent(self):
        print("\033[33m{:s}\033[0m".format('TRAINING MODEL :'))
        self.iterations = 0
        with output(output_type='dict', sort_key=lambda x: 1) as output_lines:
            while self.condition_to_stop_training():
                sum1 = 0
                sum2 = 0
                for i in range(self.M):
                    T = self.T0 + self.T1 * self.X[i] - self.Y[i]
                    sum1 += T
                    sum2 += T * self.X[i]

                self.T0 = self.T0 - self.learning_rate * (sum1 / self.M)
                self.T1 = self.T1 - self.learning_rate * (sum2 / self.M)

                self.C.append(self.cost())

                self.prev_mse = self.cur_mse
                self.cur_mse = self.cost()
                self.delta_mse = self.cur_mse - self.prev_mse

                self.iterations += 1


                if self.iterations % 100 == 0 or self.iterations == 1:
                    self.live_update(output_lines)
                    if self.live == True:
                        self.plot_all(self.po, self.pn, self.history)

            self.live_update(output_lines)

        self.RMSE_percent()
        self.MSE_percent()

        print(prYellow('SUCCESS :'))
        print(prGreen("    Applied model to data"))
        print(prYellow('RESULTS (Normalized)  :'))
        print(f'    {prCyan("Theta0           :")} {self.T0}\n    {prCyan("Theta1           :")} {self.T1}')
        print(prYellow('RESULTS (DeNormalized):'))
        print(f'    {prCyan("Theta0           :")} {self._T0}\n    {prCyan("Theta1           :")} {self._T1}')
        print("\033[33m{:s}\033[0m".format('AlGORITHM ACCURACY:'))
        print(f'    {prCyan("RMSE             : ")}{round(ftlr.RMSE, 2)} % ≈ ({ftlr.RMSE} %)')
        print(f'    {prCyan("MSE              : ")}{round(ftlr.MSE, 2)} % ≈ ({ftlr.MSE} %)')
        print(f'    {prCyan("ΔMSE             : ")}{ftlr.delta_mse}')
        print(prYellow('Storing Theta0 && Theta1:'))
        set_gradient_csv(self.output, self._T0, self._T1)
        print(prGreen("    Theta0 && Theta1 has been stored in file , open : ") + self.output)

        if self.po or self.pn or self.history:
            print(prYellow('Plotting Data:'))
            self.plot_all(self.po, self.pn, self.history, final=True)
            print(prGreen("    Data plotted successfully , open : ") + 'LR-Graph.png')

        if self.live == True:
            print(prYellow('Creating GIF image of progress:'))
            self.gifit()
            print(prGreen("    Live progress GIF created , open : ") + 'LR-Live.gif')

    def gifit(self):
        if os.path.exists('./LR-Live.gif'):
            os.remove('./LR-Live.gif')
        def sorted_ls(path):
            mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
            return list(sorted(os.listdir(path), key=mtime))

        filenames = sorted_ls('./gif')
        with imageio.get_writer('./LR-Live.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread('./gif/' + filename)
                writer.append_data(image)

    def plot_original(self):
        p1 = self.axis[0, 0]
        p1.plot(self.x, self.y, 'ro', label='data')
        x_estim = self.x
        y_estim = [denormalizeElem(self.y, self.estimatePrice(self.T0, self.T1, normalizeElem(self.x, _))) for _ in
                   x_estim]
        p1.plot(x_estim, y_estim, 'g-', label='Estimation')
        p1.set_ylabel('Price (in euro)')
        p1.set_xlabel('Mileage (in km)')
        p1.set_title('Price = f(Mileage) | Original')

    def plot_normalized(self):
        p2 = self.axis[0, 1]
        p2.plot(self.X, self.Y, 'ro', label='data')
        x_estim = self.X
        y_estim = [self.estimatePrice(self.T0, self.T1, _) for _ in x_estim]
        p2.plot(x_estim, y_estim, 'g-', label='Estimation')

        p2.set_title('Price = f(Mileage) | Normalized')

    def plot_history(self):
        p4 = self.axis[1, 1]
        p4.set_ylabel('Cost')
        p4.set_xlabel('Iterations')
        p4.set_title(f'Cost = f(iteration) | L.Rate = {self.learning_rate}')
        p4.plot([i for i in range(self.iterations)], self.C)

    def plot_show(self, p1, p2, p4, final):
        if p1 != False or p2 != False or p4 != False:
            if p1 == False:
                self.axis[0, 0].axis('off')

            if p2 == False:
                self.axis[0, 1].axis('off')

            if p4 == False:
                self.axis[1, 1].axis('off')

            self.axis[1, 0].axis('off')

            # plt.show() # in case running from Pycharm or any other editors
            imgname = f'./gif/LR-Graph-{self.iterations}.png'
            if final == True:
                imgname = f'./LR-Graph.png'

            plt.savefig(imgname)
            plt.close()

    def plot_all(self, p1, p2, p4, final=False):

        self.figure, self.axis = plt.subplots(2, 2, figsize=(10, 10))

        if p1:
            self.plot_original()
        if p2:
            self.plot_normalized()
        if p4:
            self.plot_history()

        self.plot_show(p1, p2, p4, final)


def optparse():
    """
        Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-in', action="store", dest="input", type=str, default='data.csv',
                        help='source of data file')

    parser.add_argument('--output', '-o', action="store", dest="output", type=str, default='thetas.txt',
                        help='source of data file')

    parser.add_argument('--iteration', '-it', action="store", dest="iter", type=int, default=0,
                        help='Change number of iteration. (default is Uncapped)')

    parser.add_argument('--history', '-hs', action="store_true", dest="history", default=False,
                        help='save history to futur display')

    parser.add_argument('--plotOriginal', '-po', action="store_true", dest="plot_original", default=False,
                        help="Enable to plot the original data sets")

    parser.add_argument('--plotNormalized', '-pn', action="store_true", dest="plot_normalized", default=False,
                        help="Enable to plot the normalized data sets")

    parser.add_argument('--learningRate', '-l', action="store", dest="rate", type=float, default=0.1,
                        help='Change learning coeficient. (default is 0.1)')

    parser.add_argument('--live', '-lv', action="store_true", dest="live", default=False,
                        help='Store live chnaged on gif graph')
    return parser.parse_args()


def signal_handler(sig, frame):
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    welcome = """
████████ ████████         ██       ████ ██    ██ ████████    ███    ████████          ████████  ████████  ██████   ████████  ████████  ██████   ██████  ████  ███████  ██    ██ 
██          ██            ██        ██  ███   ██ ██         ██ ██   ██     ██         ██     ██ ██       ██    ██  ██     ██ ██       ██    ██ ██    ██  ██  ██     ██ ███   ██ 
██          ██            ██        ██  ████  ██ ██        ██   ██  ██     ██         ██     ██ ██       ██        ██     ██ ██       ██       ██        ██  ██     ██ ████  ██ 
██████      ██            ██        ██  ██ ██ ██ ██████   ██     ██ ████████          ████████  ██████   ██   ████ ████████  ██████    ██████   ██████   ██  ██     ██ ██ ██ ██ 
██          ██            ██        ██  ██  ████ ██       █████████ ██   ██           ██   ██   ██       ██    ██  ██   ██   ██             ██       ██  ██  ██     ██ ██  ████ 
██          ██            ██        ██  ██   ███ ██       ██     ██ ██    ██          ██    ██  ██       ██    ██  ██    ██  ██       ██    ██ ██    ██  ██  ██     ██ ██   ███ 
██          ██            ████████ ████ ██    ██ ████████ ██     ██ ██     ██         ██     ██ ████████  ██████   ██     ██ ████████  ██████   ██████  ████  ███████  ██    ██ 

   """
    print(welcome)

    if not os.path.exists('./gif'):
        os.makedirs('./gif')

    options = optparse()
    if (options.rate < 0.0000001 or options.rate > 1):
        options.rate = 0.1
    print("\033[33m{:s}\033[0m".format('Initial Params for training model:'))
    print(prCyan('    Learning Rate    : ') + str(options.rate))
    print(prCyan('    Max iterations   : ') + "Uncapped" if str(options.iter) == "0" else "0")
    print(prCyan('    Plot Original    : ') + ('Enabled' if options.plot_original else 'Disabled'))
    print(prCyan('    Plot Normalized  : ') + ('Enabled' if options.plot_normalized else 'Disabled'))
    print(prCyan('    Plot History     : ') + ('Enabled' if options.history else 'Disabled'))
    print(prCyan('    DataSets File    : ') + options.input)
    print(prCyan('    Output File      : ') + options.output)


    ftlr = ft_linear_regression(rate=options.rate,
                                iter=options.iter,
                                input=options.input,
                                output=options.output,
                                po=options.plot_original,
                                pn=options.plot_normalized,
                                history=options.history,
                                live=options.live)
    ftlr.gradient_descent()
