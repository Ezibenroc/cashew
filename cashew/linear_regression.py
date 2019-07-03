import numpy
import pandas
import itertools
import os
from statsmodels.formula.api import ols


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


def write_regression(filename, reg_df):
    if os.path.isfile(filename):
        old_df = pandas.read_csv(filename)
        diff = set(old_df.columns) ^ set(reg_df.columns)
        if len(diff) > 0:
            raise WriteError('Incompatible columns, the following are in one dataframe but not the other: %s' % diff)
        identifier = ['cluster', 'jobid']
        old_jobs = old_df[identifier].drop_duplicates()
        new_jobs = reg_df[identifier].drop_duplicates()
        intersection = old_jobs.set_index(identifier).join(new_jobs.set_index(identifier), how='inner').reset_index()
        if len(intersection) > 0:
            raise WriteError('Statistics for the following jobs have already been computed:\n%s' % intersection)
        reg_df = reg_df[old_df.columns]  # Making sure that the columns are in the same order
        reg_df = pandas.concat([old_df, reg_df])
    else:
        id_cols = ['function', 'cluster', 'node', 'cpu', 'jobid', 'start_time']
        val_cols = list(sorted(set(reg_df.columns) - set(id_cols)))
        reg_df = reg_df[id_cols + val_cols]
    reg_df.to_csv(filename, index=False)
