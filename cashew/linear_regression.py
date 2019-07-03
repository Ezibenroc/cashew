import numpy
import pandas
import itertools
import os
import time
import datetime
from statsmodels.formula.api import ols
from .logger import logger


def powerset(iterable):
    '''
    powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    Taken from https://docs.python.org/3/library/itertools.html#itertools-recipes<Paste>
    '''
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1))


def compute_intercept(df, x, y):
    # taking the calls smallet than 100 times the smallest one
    short_calls = df[df[x] <= df[x].min() * 100]
    # removing the 5% longest calls, we have very large outliers (without this, the intercept is 3 times larger)
    short_calls = short_calls[short_calls[y] < short_calls[y].quantile(0.95)]
    return short_calls[y].mean()


def compute_reg(df, y_var, x_vars, aggregate=False):
    assert 'mnk' in x_vars
    model = '%s ~ %s + 0' % (y_var, ' + '.join(x_vars))
    df = df.copy()
    if aggregate:
        if aggregate == 'mean':
            df = df.groupby(x_vars)[[y_var]].mean().reset_index()
        elif aggregate == 'std':
            df = df.groupby(x_vars)[[y_var]].std().reset_index()
        else:
            assert False
    intercept = max(0, compute_intercept(df, 'mnk', y_var))
    df[y_var] -= intercept
    reg = ols(formula=model, data=df).fit()
    return {'intercept': intercept, **{var: reg.params[var] for var in x_vars}}


def predict(df, reg, variables):
    pred = numpy.zeros(len(df))
    pred += reg['intercept']
    for var in variables:
        pred += reg[var] * df[var]
    return pred


def compute_full_reg(df, y_var, x_vars):
    df = df.copy()
    reg_duration = compute_reg(df, y_var, x_vars, aggregate='mean')
    df['pred'] = predict(df, reg_duration, x_vars)
    df['residual'] = df[y_var] - df['pred']
    reg_residual = compute_reg(df, 'residual', x_vars, aggregate='std')
    for k, v in reg_residual.items():
        reg_duration['%s_residual' % k] = reg_residual[k]
    return reg_duration


def regression(df, y_var, x_vars):
    def get_unique(df, key):
        val = df[key].unique()
        assert len(val) == 1
        return val[0]
    reg_local = []
    for cluster in sorted(df['cluster'].unique()):
        for jobid in sorted(df['jobid'].unique()):
            tmp_node = df[(df['cluster'] == cluster) & (df['jobid'] == jobid)]
            for cpu in sorted(tmp_node['cpu'].unique()):
                tmp = tmp_node[tmp_node['cpu'] == cpu]
                reg = compute_full_reg(tmp, y_var, x_vars)
                reg['cluster'] = get_unique(tmp, 'cluster')
                reg['function'] = get_unique(tmp, 'function')
                reg['node'] = get_unique(tmp, 'node')
                reg['cpu'] = cpu
                reg['jobid'] = jobid
                reg['start_time'] = get_unique(tmp, 'start_time')
                total_flop = (2 * tmp['m'] * tmp['n'] * tmp['k']).sum()
                total_time = tmp['duration'].sum()
                reg['avg_gflops'] = total_flop / total_time * 1e-9
                reg_local.append(reg)
    return reg_local


def compute_variable_products(df, variables):
    for v_tuple in powerset(variables):
        if len(v_tuple) > 1:
            name = ''.join(v_tuple)
            df[name] = 1
            for var in v_tuple:
                df[name] *= df[var]


def read_and_stat(hdf_file, min_epoch):
    df = pandas.read_hdf(hdf_file, where=['start_time >= %d' % min_epoch])
    size = len(df)
    if size == 0:
        return size, None
    compute_variable_products(df, 'mnk')
    return size, pandas.DataFrame(regression(df, 'duration', ['mnk', 'mn', 'mk', 'nk']))


class WriteError(Exception):
    pass


def update_regression(hdf_file, output_file, overlap_time=3600*24):
    start = time.time()
    identifier = ['cluster', 'node', 'jobid', 'cpu']
    if os.path.isfile(output_file):
        old_reg = pandas.read_csv(output_file)
        max_start = old_reg['start_time'].max()
        min_epoch = max_start - overlap_time
        logger.info('File %s has statistics until %s' % (output_file, datetime.datetime.fromtimestamp(max_start)))
    else:
        old_reg = None
        min_epoch = 0
    logger.info('Computing statistics from file %s since %s' % (hdf_file, datetime.datetime.fromtimestamp(min_epoch)))
    nb_rows, new_reg = read_and_stat(hdf_file, min_epoch)
    if nb_rows == 0:
        logger.info('No new data, aborting')
        return
    if old_reg is not None:
        diff = set(old_reg.columns) ^ set(new_reg.columns)
        if len(diff) > 0:
            raise WriteError('Incompatible columns, the following are in one dataframe but not the other: %s' % diff)
        new_reg = new_reg[old_reg.columns]
        new_reg = pandas.concat([old_reg, new_reg])
        new_reg.drop_duplicates(subset=identifier, keep='first', inplace=True)
    else:
        id_cols = ['function', 'cluster', 'node', 'cpu', 'jobid', 'start_time']
        val_cols = list(sorted(set(new_reg.columns) - set(id_cols)))
        new_reg = new_reg[id_cols + val_cols]
    new_reg.sort_values(by=['start_time'] + identifier, axis=0, inplace=True)
    new_reg.to_csv(output_file, index=False, float_format='%.9e')
    stop = time.time()
    logger.info('Processed %d rows of database %s in %.02f seconds' % (nb_rows, hdf_file, stop-start))
