import io
import zipfile
import pandas
import datetime
import yaml
import sqlite3


def read_csv(archive_name, csv_name, columns=None):
    archive = zipfile.ZipFile(archive_name)
    df = pandas.read_csv(io.BytesIO(archive.read(csv_name)), names=columns)
    df.columns = df.columns.str.strip()
    return df


def read_yaml(archive_name, yaml_name):
    archive = zipfile.ZipFile(archive_name)
    return yaml.load(archive.read(yaml_name), Loader=yaml.SafeLoader)


def read_archive(archive_name, csv_name, columns=None):
    df = read_csv(archive_name, csv_name, columns)
    info = read_yaml(archive_name, 'info.yaml')
    nodes = [key for key in info if key.endswith('grid5000.fr')]
    assert len(nodes) == 1
    node = nodes[0]
    node = node[:node.index('.')]
    node = int(node[node.index('-')+1:])
    df['node'] = node
    df['cluster'] = info['cluster']
    df['jobid'] = info['jobid']
    df['cpu'] = 2*df['node'] + df['core'] % 2 - 2
    oarstat = read_yaml(archive_name, 'oarstat.yaml')
    df['start_date'] = str(datetime.datetime.fromtimestamp(oarstat['startTime']).date())
    df['index'] = -1
    for core in df['core'].unique():
        df.loc[df['core'] == core, 'index'] = range(len(df[df['core'] == core]))
    return df


def write_database(df, database_name, table_name):
    connection = sqlite3.connect(database_name)
    df.to_sql(table_name, connection, index=False, if_exists='append')
