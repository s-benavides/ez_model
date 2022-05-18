import numpy as np

# function to create a pdf and cdf from a timeseries (np_array)
# and a given set of bins (defined by x_axis, x_step and x_bound)
def pdf_ccdf(sample, xmin=-1, xmax=-1, n=30, output='pdf', edges=0):
    # This is a function that takes raw data, bins it to create a normalised
    # pdf and then uses the rank method to calculate the ccdf

    if xmin == -1:
        xmin = sample.min()
    if xmax == -1:
        xmax = sample.max()

    # get sample size
    sample = np.asarray(sample)
    sample = sample[sample>0]
    ns = sample.size

    if not sample.any():
        print('Warning: data given to pdf_ccdf are only zeros')
        # exit(0)

    # get log size of step
    e_step = (np.log10(xmax) - np.log10(xmin))/n
    # find edges of bins
    bin_edge = xmin*10**(np.arange(n+1)*e_step)
    # find centers of bins
    bin_axis = bin_edge[:-1] + np.diff(bin_edge)/2

    # initialise pdf array
    pdf = np.zeros(n)
    # go through each data point in chosen dataset and put it in the right bin
    # compare data point to the rest of the bins to find the one it fits in
    for bin in range(n):
        # search for values of s that fall between each pair of bounds
        pdf[bin] = np.sum((sample >= bin_edge[bin]) & (sample < bin_edge[bin+1]))

    # weight PDF by dividing by the bin width and the total number of measurements
    pdf = pdf/(ns*np.diff(bin_edge))

    if output == 'pdf':
        out = pdf

    elif output == 'ccdf':
        # sort data
        sample = np.sort(sample)
        # create CCDF using rank method
        # set x_rank to 0
        x_rank = np.zeros(n)

        # Find which ranked sample exceeds each bin center
        for bin in range(n):
            if not np.where(sample >= bin_axis[bin])[0].any():
                x_rank[bin] = ns
            else:
                x_rank[bin] = np.where(sample >= bin_axis[bin])[0][0]

        ccdf = 1. - x_rank/ns
        # and return it
        out = ccdf

    if edges == 0:
        return out, bin_axis

    elif edges == 1:
        return out, bin_axis, bin_edge
