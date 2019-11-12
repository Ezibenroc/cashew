import io
import zipfile
import pandas
import yaml
import lxml.etree
import os
import hashlib
from peanut import Nodes


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


def read_archive(archive_name, csv_name, columns=None):
    df = read_archive_csv(archive_name, csv_name, columns)
    info = read_yaml(archive_name, 'info.yaml')
    site = info['site']
    cluster = info['cluster']
    # changing 'dahu-42.grenoble.grid5000.fr' into '42'
    df['node'] = df['hostname'].str[len(cluster)+1:-(len(site)+len('..grid5000.fr'))]
    df['cluster'] = cluster
    df['jobid'] = info['jobid']
    core_mapping = platform_to_cpu_mapping(get_platform(archive_name))
    df['cpu'] = df.apply(lambda row: core_mapping[row.core], axis=1)
    oarstat = read_yaml(archive_name, 'oarstat.yaml')
    df['start_time'] = oarstat['startTime']
    df['index'] = -1
    for core in df['core'].unique():
        df.loc[df['core'] == core, 'index'] = range(len(df[df['core'] == core]))
    expfile = info['expfile']
    assert len(expfile) == 1
    expfile = zipfile.ZipFile(archive_name).read(expfile[0])
    expfile = expfile.split()
    expfile.sort()
    expfile = b'\n'.join(expfile)
    df['expfile_hash'] = hashlib.sha256(expfile).hexdigest()
    return df.drop(['lda', 'ldb', 'ldc', 'hostname'], axis=1)


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
    df.to_hdf(database_name, 'DATABASE', min_itemsize={'cluster': 20, 'function': 20}, **kwargs)


def read_database(database_name):
    return pandas.read_hdf(database_name)
