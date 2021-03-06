# svdawg

#### Created by Simone Longo at the University of Utah<br/>March 2020

Accessories, widgets, and graphics for singular value decomposition (SVD) analysis.

#### NOTE: This is an initial release and may be unstable. Please contact with bug information or feature requests. More functionality will be added over time.

### Installation

```bash
pip3 install svdawg
```

#### OR

Installation requires `numpy`, `seaborn`, `matplotlib`, `pandas`, and `sklearn`

In your terminal:
```
git clone https://github.com/SpacemanSpiff7/svdawg/
cd svdawg
pip3 install .
```

*Available methods*
<pre>
<b>fillnans(df, fill=0)</b>
    Parameters:
            df:   Pandas DataFrame to clean
            fill: Value to use when replacing np.nan and np.inf (default=0)

    Returns:
            Returns a copy of the input Pandas DataFrame replacing all np.nan and np.inf with the specified value


<b>generate_synthetic_data(m, n)</b>
    Generate a toy dataset with dimension m x n

    Parameters:
            m: number of rows
            n: number of columns


<b>lineplot_svs(svd, top=5)</b>
    Create lineplots of top singular values in U and V^T sorted and unsorted

    Parameters:
            svd:  A 3-ple containing the result of a SVD
            top:  Integer indicating which top singular values to include


<b>pd_scale(df)</b>
    StandardScaler transform of Pandas DataFrame, maintaining row and column labels.

    Parameters:
            df: Pandas DataFrame

    Returns:
            A scaled and labelled Pandas DataFrame


<b>pd_svd(df, labels=True)</b>
    Compute SVD on a Pandas DataFrame, maintaining row and column labels.

    Parameters:
            df: Pandas DataFrame

    Returns:
            Returns decomposition of D = U.S.V^T as labelled
            Pandas DataFrames as a 3-ple, (U, S, V^T)


<b>plot_mat(mat)</b>
    Plot a scaled matrix using red for negative and green for positive values

    Parameter:
            mat: some 2D matrix of numerical values


<b>plot_sv(svd, sv=0)</b>
    Tool for plotting U and V^T sorted by a specified singular value

    Parameters:
            svd: A 3-ple containing the result of a SVD computed by 'pd_svd'
            sv:  Integer indicating which singular value to sort by


<b>plot_svd(svd)</b>
    Tool for plotting U and V^T

    Parameters:
            svd: A 3-ple containing the result of a SVD computed by 'pd_svd'
            sv: Integer indicating which singular value to sort by
            
            
<b>plot_svs(svd, top=5)</b>
    Tool for plotting U and V^T sorted by top singular values

    Parameters:
            svd: A 3-ple containing the result of a SVD computed by 'pd_svd'
            top: Integer indicating which top singular values to sort by


<b>svd_fp(fp, header='infer', sep='\t', index_col=None)</b>
    Compute SVD directly from filepath to a table of tab-separated numerical values
    
    Parameters:
                fp:         Path to file
                header:     Specify header, see Pandas.read_csv documentation for default option
                sep:        field separator (default is tab separated). NOTE: this is different than default Pandas behavior
                index_col:  specify if dataframe has an existing index (see default Pandas.read_csv documentation)

    Returns:
                A tuple containing the input dataframe and the result of a SVD on the data
                Tuple Contents: (Pandas.DataFrame, (U, S, V^T))


<b>svd_overview(data, top=3, scale=True)</b>
    Display original data with line plots of top singular values from V^T and U

    Parameters:
            data:   untransformed dataframe
            top:    top n singular values to plot
            scale:  Preprocess data before SVD (boolean)


<b>svdfilter(svd, noise=[0])</b>
    Tool for filtering a singular value and reconstructing a data set

    Parameters:
            svd:    A 3-ple containing the result of a SVD
            noise:  A list enumerating the singular values to set to 0

    Returns:
            Reconstruction of the filtered dataset as a NumPy array
</pre>

Example:
```python
import svdawg as sv

toydata = sv.generate_synthetic_data(100,10)
toydata.head()
```

```
Out[]:
          0         1         2         3         4         5         6         7         8         9
0  1.000000  0.809017  0.309017 -0.309017 -0.809017 -1.000000 -0.809017 -0.309017  0.309017  0.809017
1  0.998027  0.844328  0.368125 -0.248690 -0.770513 -0.998027 -0.844328 -0.368125  0.248690  0.770513
2  0.992115  0.876307  0.425779 -0.187381 -0.728969 -0.992115 -0.876307 -0.425779  0.187381  0.728969
3  0.982287  0.904827  0.481754 -0.125333 -0.684547 -0.982287 -0.904827 -0.481754  0.125333  0.684547
4  0.968583  0.929776  0.535827 -0.062791 -0.637424 -0.968583 -0.929776 -0.535827  0.062791  0.637424
```

```python
# Visualize the data
sv.plot_mat(toydata)
```

![Visualization of 'toydata' DataFrame](https://github.com/SpacemanSpiff7/images/blob/master/toydata_vis.png)


#### Calculate SVD
```python
# Generate SVD results to use with other methods
svd = sv.pd_svd(toydata)

# Visualize results
sv.plot_svd(svd)
```

![Visualization of 'toydata' SVD](https://github.com/SpacemanSpiff7/images/blob/master/plot_svd_example.png)

```python
# Examine top singular values
sv.lineplot_svs(svd, top=4)
```

![Top 4 Singular Values of 'toydata'](https://github.com/SpacemanSpiff7/images/blob/master/lineplot_svs_example.png)

```python
# Filter out first singular value
sv.plot_mat(sv.svdfilter(svd))
```

![Filtered 1st SV](https://github.com/SpacemanSpiff7/images/blob/master/svdfilter_example.png)

```python
# Plot SVD sorted by top singular values
sv.plot_svs(svd, top=4)
```

![Plotted Singular Values](https://github.com/SpacemanSpiff7/images/blob/master/plot_svs_example.png)

```python
# Or just plot whichever one you choose
sv.plot_sv(svd, sv=0)
```

![Sorted by 0th SV](https://github.com/SpacemanSpiff7/images/blob/master/single_sv.png)


```python
# Quickly visualize Singular values in the context of your original data
sv.svd_overview(toydata, top=3)
```

![SVD Overview](https://github.com/SpacemanSpiff7/images/blob/master/Screen%20Shot%202020-03-20%20at%208.05.44%20AM.png)
