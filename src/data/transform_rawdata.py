import os
import pandas as pd
import datetime as dt
import json
from statsmodels.tsa.stattools import adfuller
from src.config import Config


def create_product2idx(df:pd.DataFrame, colname="product_name", save=False, new_column=False, path=None):
    """
    Create indexes from column.unique values and replaces column.values with indexes.
    Instead of replacement can create additional column with indexes. Provide new_column name for that
    By save=True saves two dictionaries (idx2product and product2idx) in textfiles.
    """
    prod2idx = {}
    idx2prod = {}
    for i,name in enumerate(df[colname].unique()):
        prod2idx[name] = i
        idx2prod[i] = name
    
    if path is not None:
        p1 = os.path.join(path,'prod2idx.json')
        p2 = os.path.join(path,'idx2prod.json')
    else:
        p1 = Config.get_path2file('prod2idx.json', report=True)
        p2 = Config.get_path2file('idx2prod.json', report=True)

    if save:
        with open(p1, 'w') as fp:
            json.dump(prod2idx, fp)
        with open(p2, 'w') as fp:
            json.dump(idx2prod, fp)
    
    prod2idx_list = []
    for idx,val in df[colname].iteritems():
        new_name = prod2idx[val]
        prod2idx_list.append(new_name)
    if new_column:
        colname = new_column    
    df[colname] = prod2idx_list

def transform_rawdata(fname=Config.FNAME, path=None):
    """
    Read raw dataset.
    Filter out unnecessary columns and values.
    Save result in folder data/processed
    """
    if path is not None:
        df = pd.read_csv(path)
    else:    
        df = pd.read_csv(Config.get_path2file(fname, rawdata=True))
    df = df.loc[df['Diff Load Due Week'] == -1]
    df = df.drop(['BM', 'SAP Order Type', 'Consi Flag', 'Sold-to SAP Cust No',
        'Sold-to Business Name', 'Sold-to Sales Office', 'Sold-to Sales Org',
        'SP SAP Matnr', 'RFP SAP Matnr', 'PL',
        'Division Short Desc', 'Diff Load Due Week',
        'ForecastsAndOrders'], axis=1)
    df.rename(columns={'Delivery Week Due': 'date', 'BillingsAndBacklogs': 'billings', 'Product Name': 'product_name'}, inplace=True)
    create_product2idx(df, new_column="product", save=True)
    df = df.drop("product_name", axis=1)
    
    # Add missing fiscal weeks
    missing_fiscal_weeks = [201429, 201613,201625, 201728,201748,201826]
    missing_vals_list = []
    for j in missing_fiscal_weeks:
        for i in df['product'].unique():
            missing_vals_dict = {}
            missing_vals_dict['date'] = j
            missing_vals_dict['billings'] = 0
            missing_vals_dict['product'] = i
            missing_vals_list.append(missing_vals_dict)
    df_missing = pd.DataFrame(missing_vals_list)
    df = pd.concat([df,df_missing],axis=0, ignore_index=True).sort_values('date')
    if path is not None:
        return df
    df.to_csv(Config.get_path2file('df.csv', processeddata=True))


def create_df_with_cluster(path2data=Config.get_path2file('df.csv', processeddata=True),
                           path2clusters=Config.get_path2file('SOM_8Neurons_15Epochs_Summary_pickle', external=True),
                            path2json=Config.get_path2file('prod2idx.json', report=True)):
    """
    Read processed dataset.
    Read cluster SOM_8Neurons_15Epochs_Summary_pickle from https://github.com/sebastianachter/Clustering_for_data_generation
    """
    df = pd.read_csv(path2data, index_col=0)
    pickle = pd.read_pickle(path2clusters)
    members = pickle['Members']
    with open(path2json, "r") as f:
        prod2idx = json.load(f)

    members2idx = {}
    idx2cluster = {}
    for k,m in members.items():
        idxlist = [prod2idx[i] for i in m]
        members2idx[k] = idxlist
        for i in idxlist:
            idx2cluster[i] = k

    pickle['products'] = members2idx
    s = pd.DataFrame(pickle['products'].values)[0].explode()
    dd = pd.DataFrame(s).reset_index()
    df = df.merge(dd, left_on='product', right_on=dd[0])
    return df

def create_list_of_dfclusters_from_df(df, clustercol='index', clusters=range(8)):
    return [df.loc[df[clustercol] == i] for i in clusters]

def get_clusters_from_dataset():
    df = create_df_with_cluster()
    df.drop([0],axis=1, inplace=True)
    listofdfs = create_list_of_dfclusters_from_df(df)
    for i,v in enumerate(listofdfs):
        fname = f"cluster_{i}.csv"
        v.to_csv(Config.get_path2file(fname, processeddata=True))


def read_cluster_data(clrn):
    filename = 'cluster_'+clrn
    path2clr = Config.get_path2file(filename, processeddata=True)
    df = pd.read_csv(path2clr, index_col=0)
    df['time_idx'] = [dt.datetime.strptime(str(x)+'-1', "%Y%W-%w") for x in df['date']]
    df_pivot = pd.pivot_table(df,
                             values='billings',
                             index='time_idx',
                             columns='product',
                             fill_value=0)
    return df_pivot

def get_idxs_high_autocorrelation(df, lag=1):
    autocorr_dict = {}
    for col in df.columns:
        autocorr = df[col].autocorr(lag=lag)
        if autocorr > 0.5 or autocorr < -0.5:
            autocorr_dict[col] = autocorr
    return autocorr_dict

def create_benchmark(path=None):
    if path is not None:
        df = pd.read_csv(path, index_col=0)
    else:
        df = pd.read_csv(Config.get_path2file('df.csv', processeddata=True), index_col=0)
    df['date'] = df['date'].astype('int')
    df['time_idx'] = [dt.datetime.strptime(str(x)+'-1', "%Y%W-%w") for x in df['date']]
    df = pd.pivot_table(df,
                        values='billings',
                        index='time_idx',
                        columns='product',
                        fill_value=0)
    benchidx = []
    for col in df.columns:
        adf = adfuller(df[col].values)
        autocorr1 = df[col].autocorr(lag=1)
        autocorr12 = df[col].autocorr(lag=12)
        autocorr24 = df[col].autocorr(lag=24)
        if adf[1] < 0.05 and (autocorr1 > 0.5 or autocorr12 > 0.5 or autocorr24 >0.5):
            benchidx.append(col)
    benchmark = df[benchidx].copy()
    if path is not None:
        return benchmark
    benchmark.to_csv(Config.get_path2file('benchmark.csv', processeddata=True))

def data_preparation():
    transform_rawdata()
    get_clusters_from_dataset()
    create_benchmark()
