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
    # taking the calls smallet than 3 times the smallest one
    short_calls = df[df[x] <= df[x].min() * 3]
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
    result = {'intercept': intercept, **{var: reg.params[var] for var in x_vars}}
    result.update({('tvalue_%s' % var): reg.tvalues[var] for var in x_vars})
    return result


def predict(df, reg, variables):
    pred = numpy.zeros(len(df))
    pred += reg['intercept']
    for var in variables:
        pred += reg[var] * df[var]
    return pred


def compute_full_reg(df, y_var, x_vars):
    reg_duration = compute_reg(df, y_var, x_vars, aggregate='mean')
    df['pred'] = predict(df, reg_duration, x_vars)
    df['residual'] = df[y_var] - df['pred']
    reg_residual = compute_reg(df, 'residual', x_vars, aggregate='std')
    for k, v in reg_residual.items():
        reg_duration['%s_residual' % k] = reg_residual[k]
    return reg_duration

def get_unique(df, key):
    val = df[key].unique()
    assert len(val) == 1
    return val[0]


def compute_dgemm_reg(df):
    df = df.copy()
    compute_variable_products(df, 'mnk')
    reg = compute_full_reg(df, 'duration', ['mnk', 'mn', 'mk', 'nk'])
    total_flop = (2 * df['m'] * df['n'] * df['k']).sum()
    total_time = df['duration'].sum()
    reg['mean_gflops'] = total_flop / total_time * 1e-9
    reg['function'] = get_unique(df, 'function')
    return reg


def compute_monitoring_stat(df, time_after_start=120, time_window=240):
    start = get_unique(df, 'start_exp')
    stop = get_unique(df, 'stop_exp')
    if stop - start < time_after_start + 2*time_window:
        raise ValueError('Experiment was too short, cannot compute monitoring values')
    tmp = df[(df['timestamp'] > start + time_after_start) &
              (df['timestamp'] < start + time_after_start + time_window)]
    freq = tmp[tmp['kind'] == 'frequency']['value']
    temp = tmp[tmp['kind'] == 'temperature']
    if temp['core'].min() < 0:  # we have values for the whole CPU, let's use that instead of the per-core values
        temp = temp[temp['core'] < 0]
    temp = temp['value']
    result = {
        'mean_frequency': freq.mean(),
        'std_frequency': freq.std(),
        'mean_temperature': temp.mean(),
        'std_temperature': temp.std(),
    }
    power_cpu  = tmp[tmp['kind'] == 'power_cpu']['value']
    power_dram = tmp[tmp['kind'] == 'power_dram']['value']
    if len(power_cpu) > 0:
        result.update({
            'mean_power_cpu': power_cpu.mean(),
            'std_power_cpu': power_cpu.std(),
            'mean_power_dram': power_dram.mean(),
            'std_power_dram': power_dram.std(),
    })
    return result


def regression(df, reg_func):
    reg_local = []
    for cluster in sorted(df['cluster'].unique()):
        tmp_cluster = df[df['cluster'] == cluster]
        for jobid in sorted(tmp_cluster['jobid'].unique()):
            tmp_job = tmp_cluster[tmp_cluster['jobid'] == jobid]
            for node in sorted(tmp_job['node'].unique()):
                tmp_node = tmp_job[tmp_job['node'] == node]
                for cpu in sorted(tmp_node['cpu'].unique()):
                    tmp = tmp_node[tmp_node['cpu'] == cpu]
                    reg = reg_func(tmp)
                    reg['cluster'] = get_unique(tmp, 'cluster')
                    reg['node'] = get_unique(tmp, 'node')
                    reg['expfile_hash'] = get_unique(tmp, 'expfile_hash')
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


def read_and_stat(hdf_file, min_epoch, max_epoch=None, conditions=[]):
    df = pandas.read_hdf(hdf_file, where=['start_time >= %d' % min_epoch]+conditions)
    size = len(df)
    if size == 0:
        return size, None
    if 'start_exp' in df.columns:  # monitoring data
        return size, pandas.DataFrame(regression(df, compute_monitoring_stat))
    else:  # performance data
        return size, pandas.DataFrame(regression(df, compute_dgemm_reg))


class WriteError(Exception):
    pass


def update_regression(hdf_file, output_file, overlap_time=3600*12, conditions=[]):
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
    nb_rows, new_reg = read_and_stat(hdf_file, min_epoch, conditions=conditions)
    if nb_rows == 0:
        logger.info('No new data, aborting')
        return
    if old_reg is not None:
        diff = set(old_reg.columns) ^ set(new_reg.columns)
        if len(diff) > 0:
            logger.warning('Some columns were in one dataframe but not the other: %s' % diff)
        new_reg = pandas.concat([old_reg, new_reg])
        new_reg.drop_duplicates(subset=identifier, keep='first', inplace=True)
    else:
        id_cols = ['cluster', 'node', 'cpu', 'jobid', 'start_time', 'expfile_hash']
        if 'function' in new_reg.columns:
            id_cols.append('function')
        val_cols = list(sorted(set(new_reg.columns) - set(id_cols)))
        new_reg = new_reg[id_cols + val_cols]
    new_reg.sort_values(by=['start_time'] + identifier, axis=0, inplace=True)
    new_reg.to_csv(output_file, index=False, float_format='%.9e')
    stop = time.time()
    logger.info('Processed %d rows of database %s in %.02f seconds' % (nb_rows, hdf_file, stop-start))
