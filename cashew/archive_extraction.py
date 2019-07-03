import io
import zipfile
import pandas
import yaml
import lxml.etree
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
    keys = list(set(platform[0].keys()) - {'PU', 'Core'})
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
    nodes = [key for key in info if key.endswith('grid5000.fr')]
    assert len(nodes) == 1
    node = nodes[0]
    node = node[:node.index('.')]
    node = int(node[node.index('-')+1:])
    df['node'] = node
    df['cluster'] = info['cluster']
    df['jobid'] = info['jobid']
    core_mapping = platform_to_cpu_mapping(get_platform(archive_name))
    df['cpu'] = df.apply(lambda row: core_mapping[row.core], axis=1)
    oarstat = read_yaml(archive_name, 'oarstat.yaml')
    df['start_time'] = oarstat['startTime']
    df['index'] = -1
    for core in df['core'].unique():
        df.loc[df['core'] == core, 'index'] = range(len(df[df['core'] == core]))
    return df


def write_database(df, database_name, **kwargs):
    df.to_hdf(database_name, 'DATABASE', **kwargs)


def read_database(database_name):
    return pandas.read_hdf(database_name)
