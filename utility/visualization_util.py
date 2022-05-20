import matplotlib.pyplot as plt


def plot_data(serires, titile, ylable, xlabel, legend):
    fig, ax = plt.subplots()
    plt.plot(serires)
    plt.title(titile)
    plt.ylabel(ylable)
    plt.xlabel(xlabel)
    ax.legend(legend)
    # ax.legend(["BTC/Euro monthly"])
