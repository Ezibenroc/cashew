import io
import zipfile
import pandas
import numpy
import yaml
import lxml.etree
import os
import re
import hashlib
from peanut import Nodes
from .logger import logger

modes = {
    'performance': 'result.csv',
    'monitoring': 'monitoring.csv',
}


def read_archive_csv(archive_name, csv_name, columns=None):
    archive = zipfile.ZipFile(archive_name)
    df = pandas.read_csv(io.BytesIO(archive.read(csv_name)), names=columns)
    df.columns = df.columns.str.strip()
    return df


def read_yaml(archive_name, yaml_name):
    archive = zipfile.ZipFile(archive_name)
    return yaml.load(archive.read(yaml_name), Loader=yaml.SafeLoader)


def get_platform(archive_name, filename='topology.xml'):
    archive = zipfile.ZipFile(archive_name)
    files = [f.filename for f in archive.filelist if filename in f.filename]
    topologies = [Nodes.get_all_cores(lxml.etree.fromstring(archive.read(f))) for f in files]
    ref = topologies[0]
    for topo in topologies:
        assert topo == ref
    return ref


def platform_to_cpu_mapping(platform):
    cpus = {}
    mapping = {}
    keys = list(set(platform[0].keys()) - {'NUMANode', 'PU', 'Core'})
    CPU_id = 0
    for PU, info in sorted(platform.items()):
        identifier = tuple(info[k] for k in keys)
        try:
            cpus[identifier]
        except KeyError:
            cpus[identifier] = CPU_id
            CPU_id += 1
        mapping[PU] = cpus[identifier]
    return mapping


def read_archive_csv_enhanced(archive_name, csv_name, columns=None, dropna=False):
    df = read_archive_csv(archive_name, csv_name, columns)
    if dropna:
        old_len = len(df)
        df.dropna(inplace=True)
        new_len = len(df)
        if new_len < old_len:
            logger.warning(f'File {csv_name} from archive {archive_name} contained missing value, dropped {old_len-new_len} rows')
            int_cols = ['node', 'cpu', 'core', 'm', 'n', 'k', 'index']
            for col in set(int_cols) & set(df.columns):
                df[col] = df[col].astype(int)
    info = read_yaml(archive_name, 'info.yaml')
    site = info['site']
    cluster = info['cluster']
    try:  # retro-compatibility...
        df['hostname']
    except KeyError:
        nodes = [key for key in info if key.endswith('grid5000.fr')]
        assert len(nodes) == 1
        df['node'] = nodes[0]
    else:
        df['node'] = df['hostname']
    # changing 'dahu-42.grenoble.grid5000.fr' into '42'
    df['node'] = df['node'].str[len(cluster)+1:-(len(site)+len('..grid5000.fr'))].astype(int)
    df['cluster'] = cluster
    df['jobid'] = info['jobid']
    oarstat = read_yaml(archive_name, 'oarstat.yaml')
    df['start_time'] = oarstat['startTime']
    expfile = info['expfile']
    assert len(expfile) == 1
    expfile = zipfile.ZipFile(archive_name).read(expfile[0])
    expfile = expfile.split()
    expfile.sort()
    expfile = b'\n'.join(expfile)
    df['expfile_hash'] = hashlib.sha256(expfile).hexdigest()
    return df


def read_performance(archive_name, columns=None):
    '''
    Read the durations of a BLAS calibration in an archive.
    '''
    csv_name = 'result.csv'
    df = read_archive_csv_enhanced(archive_name, csv_name, columns=columns, dropna=True)
    core_mapping = platform_to_cpu_mapping(get_platform(archive_name))
    df['cpu'] = df.apply(lambda row: core_mapping[row.core], axis=1)
    df['index'] = -1
    for core in df['core'].unique():
        df.loc[df['core'] == core, 'index'] = range(len(df[df['core'] == core]))
    columns = ['function', 'm', 'n', 'k', 'timestamp', 'duration', 'core', 'node',
       'cluster', 'jobid', 'cpu', 'start_time', 'index', 'expfile_hash']
    return df[columns]


def my_melt(df, pattern, new_name, idcol):
    reg = re.compile(pattern)
    result = []
    for col in df.columns:
        match = reg.fullmatch(col)
        if match is None:
            continue
        if len(match.groups()) != 1:
            raise ValueError('Column "%s" matched with pattern "%s" but with %d groups' % (col, pattern,
                len(match.groups())))
        group = match.groups()[0]
        tmp = df[idcol].copy()
        tmp[new_name] = df[col]
        tmp['group'] = int(group)
        result.append(tmp)
    return pandas.concat(result)


def read_monitoring(archive_name, columns=None):
    '''
    Read the durations of a BLAS calibration in an archive.
    '''
    csv_name = 'monitoring.csv'
    df = read_archive_csv_enhanced(archive_name, csv_name, columns=columns)
    df['timestamp'] = pandas.to_datetime(df['timestamp'])
    core_mapping = platform_to_cpu_mapping(get_platform(archive_name))
    columns = ['timestamp', 'cluster', 'node', 'jobid', 'start_time', 'expfile_hash']
    frequency   = my_melt(df, 'frequency_core_([0-9]+)', 'value', columns)
    frequency = frequency[frequency['group'].isin(core_mapping)]
    frequency['value'] *= 1e-9  # Hz → GHz
    monitoring_values = [(frequency, 'frequency')]
    try:
        temperature = my_melt(df, 'temperature_core_([0-9]+)', 'value', columns)
    except ValueError:
        logger.warning('No core temperature available')
    else:
        temperature = temperature[temperature['group'] <= max(core_mapping.keys())]
        monitoring_values.append((temperature, 'temperature'))
    for frame, val in monitoring_values:
        frame['cpu'] = frame.apply(lambda row: core_mapping[row.group], axis=1)
        frame['core'] = frame['group']
        frame['kind'] = val
    result = pandas.concat([v[0] for v in monitoring_values])
    try:
        temperature_cpu = my_melt(df, 'temperature_cpu_([0-9]+)', 'value', columns)
    except ValueError:
        logger.warning('No CPU temperature available')
        key = None
        if 'temperature_acpitz' in df.columns:
            key = 'temperature_acpitz'
        elif 'temperature_k10temp_Tdie' in df.columns:
            key = 'temperature_k10temp_Tdie'
        if key:
            temperature_cpu = df[columns + [key]].copy()
            temperature_cpu['cpu'] = 0
            temperature_cpu['core'] = 0
            temperature_cpu['kind'] = 'temperature'
            temperature_cpu['value'] = temperature_cpu[key]
            temperature_cpu.drop(key, axis=1, inplace=True)
            result = pandas.concat([result, temperature_cpu])
    else:
        temperature_cpu['cpu'] = temperature_cpu['group']
        temperature_cpu['core'] = -1
        temperature_cpu['kind'] = 'temperature'
        result = pandas.concat([result, temperature_cpu])
    try:
        power_cpu  = my_melt(df, 'power_package-([0-9]+)', 'value', columns)
        power_dram = my_melt(df, 'power_package-([0-9]+)_dram', 'value', columns)
    except ValueError:
        logger.warning('No CPU power available')
    else:
        for frame, val in [(power_cpu, 'power_cpu'), (power_dram, 'power_dram')]:
            frame['cpu'] = frame['group']
            frame['core'] = -1
            frame['kind'] = val
            result = pandas.concat([result, frame])
    info = read_yaml(archive_name, 'info.yaml')
    timestamps = info['timestamp']
    for step in ['start', 'stop']:
        result[f'{step}_exp'] = pandas.to_datetime(timestamps['run_exp'][step]).timestamp()
    result['timestamp'] = result['timestamp'].astype(numpy.int64) / 10 ** 9
    result.drop('group', axis=1, inplace=True)
    return result


def write_database(df, database_name, **kwargs):
    if os.path.exists(database_name):
        def get_unique(df, key):
            val = df[key].unique()
            assert len(val) == 1
            return val[0]
        jobid = get_unique(df, 'jobid')
        cluster = get_unique(df, 'cluster')
        tmp = pandas.read_hdf(database_name, 'DATABASE', where=['jobid=%d' % jobid, 'cluster=%s' % cluster])
        if len(tmp) > 0:
            raise ValueError('Job %d from cluster %s already exists in database %s' % (jobid, cluster, database_name))
    min_size = {'cluster': 20}
    if 'function' in df.columns:
        min_size['function'] = 20
    df.to_hdf(database_name, 'DATABASE', min_itemsize=min_size, **kwargs)


def read_database(database_name):
    return pandas.read_hdf(database_name)
