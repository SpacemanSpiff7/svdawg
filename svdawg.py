import numpy as np
import sys
import time
import argparse
import math 
import scipy.linalg as sclin
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


"""
Reference:
https://www.d.umn.edu/~mhampton/m4326svd_example.pdf
"""


def svd_fp(fp, header='infer', delim='\t'):
    df = pd.read_csv(fp, header=header, sep=delim)
    try:
        df = df.astype(float)
    except (ValueError, TypeError):
        print('Data must be numeric')
        exit(1)
    return pd_svd(df)


def test_equality(df1, df2):
    eq = np.isclose(df1, df2)
    if False in eq:
        return "WARNING: Original and reconstructed matrices are different."
    return "Original and reconstructed are equal"


def plot_mat(mat):
    color_palette = sns.color_palette("PiYG", 50)
    sns.heatmap(mat, cmap=color_palette)
    plt.show()

def generate_synthetic_data(): # noise=None):
    pi2 = math.pi * 2
    x1 = np.arange(0, pi2, pi2/50)
    x2 = np.arange(0,pi2, pi2/1000)
    y1 = np.sin(x1)
    y2 = np.sin(x2)
    y3 = np.cos(x1)
    y4 = np.cos(x2)
    mat = np.outer(y1, y2)
    mat2 = np.outer(y3, y4)
    return pd.DataFrame(mat + mat2)

def sort_by_top_value(svd_results):
    U, sigma, vt = svd_results
    reconstructed = pd.DataFrame(U @ sigma @ vt)
    udf = pd.DataFrame(U)
    vtdf = pd.DataFrame(vt)
    vtdf = vtdf.sort_values(by=0, axis=1)
    udf = udf.sort_values(by=0)
    reconstructed = reconstructed.iloc[udf.index.to_list(), vtdf.columns.to_list()]
    return reconstructed

def pd_svd(df):
    U, s, vt = np.linalg.svd(df, full_matrices=False)
    U = pd.DataFrame(U, index=df.index, columns=df.columns)
    s = np.diag(s)
    vt = pd.DataFrame(vt, index=df.columns, columns=df.columns)
    return U, s, vt

def pd_scale(df):
    n = StandardScaler().fit_transform(df)
    return pd.DataFrame(n, index=df.index, columns=df.columns)

def pd_svd(df, labels=True):
    U, s, vt = np.linalg.svd(df, full_matrices=False)
    if labels:
        if len(df.index) > len(df.columns):
            vtcols = df.columns
            vtidx = df.columns
            ucols = df.columns
            uidx = df.index
        else:
            vtcols = df.columns
            vtidx = df.index
            uidx = df.index
            ucols = df.index
        U = pd.DataFrame(U, index=uidx, columns=ucols)
        #s = np.diag(s)
        vt = pd.DataFrame(vt, index=vtidx, columns=vtcols)
    else:
        U = pd.DataFrame(U)
        vt = pd.DataFrame(vt)
    return U, s, vt

def plot_mat_ax(mat, ax):
    color_palette = sns.color_palette("PiYG", 50)
    sns.heatmap(mat, cmap=color_palette, ax=ax)
    #plt.show()

def plot_svs(svd, top=5):
    fig, ax = plt.subplots(2, top)
    for i in range(top):
        plot_mat_ax(svd[0].sort_values(by=svd[0].columns[i]), ax[0,i])
        #ax[0,i].set_aspect('auto')
        plot_mat_ax(svd[2].sort_values(by=svd[2].index[i], axis = 1), ax[1,i])
        #ax[1,i].set_aspect(aspect=1)

        
def plot_svd(svd, sv=0):
    return plot_mat(svd[0].sort_values(by=sv)), plot_mat(svd[2].sort_values(by=sv, axis=1))
    
def svdfilter(svd, noise=[0]):
    U, s, vt = svd
    for i in noise:
        s[i,i] = 0
    return U @ s @ vt

def plotlines(data, ax, orient='wide'):
    if orient=='wide':
        ax.scatter(range(len(data)), data, c='red')
        ax.scatter(range(len(data)), sorted(data), c='blue')
    elif orient=='square':
        ax.scatter(range(len(data)), data, c='red')
        ax.scatter(range(len(data)), sorted(data), c='blue')
        ax.set_aspect('equal')
    else:
        ax.scatter(data, range(len(data)), c='red')
        ax.scatter(sorted(data), range(len(data)), c='blue')


def lineplot_svs(svd, top=5):
    svd = list(svd)
    if isinstance(svd[0], pd.DataFrame):
        svd[0] = np.array(svd[0])
        svd[2] = np.array(svd[2])
    topU = [svd[0][:,num] for num in range(top)]
    topvt = [svd[2][num,:] for num in range(top)]
    if len(svd[0]) > len(svd[0][0]): # if U has more columns than rows
        # vT will be wide
        fig, ax = plt.subplots(top, 2)
        for i, sv in enumerate(topU):
            plotlines(sv, ax[i][0], orient='wide')
        for i, sv in enumerate(topvt):
            plotlines(sv, ax[i][1])
    else:
        # U will be long
        fig, ax = plt.subplots(2, top)
        for i, sv in enumerate(topU):
            plotlines(sv, ax[0][i], orient='long')
        for i, sv in enumerate(topvt):
            plotlines(sv, ax[1][i])


if __name__=="__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', action='store', dest='filepath', help='path to file with tab separated numeric values')
    parser.add_argument('-d', '-delimiter', action='store', dest='delim', help='value separator (default is tab/indent)')
    parser.add_argument('-i', '--header', action='store_true', dest='header', help='use this flag if input file has a header row')
    parser.add_argument('--test', action='store_true', dest='testrun')
    args = parser.parse_args()
    #print(args)
    if args.testrun:
        plot_mat(sort_by_top_value(svd_df(generate_synthetic_data())))
        exit(0)
    if args.header:
        headerval = 'infer'
    else:
        headerval = None
    results = svd_fp(args.filepath, header=headerval, delim=args.delim)
    plot_mat(sort_by_top_value(results))
