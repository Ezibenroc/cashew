import io
import zipfile
import pandas
import datetime
import yaml
import os
import shutil
import sqlite3


def read_archive_csv(archive_name, csv_name, columns=None):
    archive = zipfile.ZipFile(archive_name)
    df = pandas.read_csv(io.BytesIO(archive.read(csv_name)), names=columns)
    df.columns = df.columns.str.strip()
    return df


def read_yaml(archive_name, yaml_name):
    archive = zipfile.ZipFile(archive_name)
    return yaml.load(archive.read(yaml_name), Loader=yaml.SafeLoader)


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
    df['cpu'] = 2*df['node'] + df['core'] % 2 - 2
    oarstat = read_yaml(archive_name, 'oarstat.yaml')
    df['start_date'] = str(datetime.datetime.fromtimestamp(oarstat['startTime']).date())
    df['index'] = -1
    for core in df['core'].unique():
        df.loc[df['core'] == core, 'index'] = range(len(df[df['core'] == core]))
    return df


def write_sql(df, database_name, table_name):
    connection = sqlite3.connect(database_name)
    df.to_sql(table_name, connection, index=False, if_exists='append')


def read_sql(database_name, table_name):
    connection = sqlite3.connect(database_name)
    return pandas.read_sql('select * from %s' % table_name, connection)


def write_csv(df, database_name, table_name):
    if os.path.isfile(database_name):
        with open(database_name) as f:
            header = f.readline()
        header = [h.strip() for h in header.split(',')]
        with open(database_name, 'a') as db_file:
            df = df[header]  # eventual reordering of the columns, to match the already existing CSV
            df.to_csv(db_file, index=False, header=False)
    else:
        df.to_csv(database_name, index=False)


def read_csv(database_name, table_name):
    return pandas.read_csv(database_name)


NESTED_COLUMNS = [
        'function',
        'cluster',
        'node',
        'core',
        'jobid'
]


def write_nested_csv(df, database_name, table_name, columns=NESTED_COLUMNS):
    if len(columns) > 0:
        col = columns[0]
        for val in sorted(df[col].unique()):
            directory = os.path.join(database_name, str(val))
            try:
                os.makedirs(directory)
            except FileExistsError:
                pass
            tmp = df[df[col] == val].drop(col, axis=1)
            write_nested_csv(tmp, directory, table_name, columns[1:])
    else:
        write_csv(df, os.path.join(database_name, 'data.db'), table_name)


def read_nested_csv(database_name, table_name, columns=NESTED_COLUMNS):
    if len(columns) > 0:
        all_df = []
        for col in os.listdir(database_name):
            df = read_nested_csv(os.path.join(database_name, col), table_name, columns[1:])
            try:
                col = int(col)
            except ValueError:
                pass
            df[columns[0]] = col
            all_df.append(df)
        return pandas.concat(all_df)
    else:
        return read_csv(os.path.join(database_name, 'data.db'), table_name)


def write_database(df, database_name, table_name, compress=False, how='sql'):
    func = {'sql': write_sql, 'csv': write_csv, 'nested_csv': write_nested_csv}[how]
    name_in_archive = 'DATABASE'
    if compress and os.path.isfile(database_name):  # first, we need to decompress the old version
        content = zipfile.ZipFile(database_name).read(name_in_archive)
        with open(database_name, 'wb') as f:
            f.write(content)
    func(df, database_name, table_name)
    if compress:
        shutil.move(database_name, name_in_archive)
        f = zipfile.ZipFile(database_name, 'w', zipfile.ZIP_DEFLATED)
        f.write(name_in_archive, name_in_archive)
        os.remove(name_in_archive)


def read_database(database_name, table_name, compress=False, how='sql'):
    func = {'sql': read_sql, 'csv': read_csv, 'nested_csv': read_nested_csv}[how]
    name_in_archive = 'DATABASE'
    if compress and os.path.isfile(database_name):  # first, we need to decompress the old version
        content = zipfile.ZipFile(database_name).read(name_in_archive)
        with open(database_name, 'wb') as f:
            f.write(content)
    return func(database_name, table_name)
