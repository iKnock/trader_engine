import matplotlib.pyplot as plt


def plot_data(serires, titile, xlabel, ylable, legend):
    fig, ax = plt.subplots()
    plt.plot(serires)
    plt.title(titile)
    plt.xlabel(xlabel)
    plt.ylabel(ylable)
    ax.legend(legend)
    # ax.legend(["BTC/Euro monthly"])


def plot_data_many(seriess, titile, xlabel, ylable, legend):
    fig, ax = plt.subplots()
    for series in seriess:
        plt.plot(series)

    plt.title(titile)
    plt.xlabel(xlabel)
    plt.ylabel(ylable)
    ax.legend(legend)
    # ax.legend(["BTC/Euro monthly"])
