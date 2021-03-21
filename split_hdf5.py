import sys
import pandas
import datetime
import time
import os

'''
This short script is intended to split a large HDF5 file into several (smaller) HDF5 files. It assumes that the input
file has one dataframe with a 'start_time' column representing a date as a Unix epoch.
The script takes as input the minimum year (when was the very first experiment done), as we do not want to load the
whole file in memory. It also takes as input the time interval, as a number of months.
'''

class NoData(Exception):
    pass

def extract_from_file(input_name, output_name, first_epoch, last_epoch):
    t = time.time()
    df = pandas.read_hdf(input_name, where=[f'start_time >= {first_epoch}', f'start_time < {last_epoch}'])
    if len(df) == 0:
        raise NoData()
    df.to_hdf(output_name, 'DATABASE', complib='zlib', complevel=9, format='table',
              data_columns=['function', 'cluster', 'node', 'jobid', 'start_time']
            )
    print(f'Wrote {len(df)} rows in file {output_name}, took {time.time()-t:.0f} seconds')


def extract_all(input_name, first_year, period):
    filename, file_extension = os.path.splitext(input_name)
    thisyear = datetime.date.today().year
    dates = list(pandas.date_range(start=f'{first_year}/01/01', end=f'{thisyear+1}/01/01', freq=f'{period}MS'))
    for start, stop in zip(dates[:-1], dates[1:]):
        start_str = str(start.date())
        stop_str = str(stop.date() - datetime.timedelta(days=1))
        start_epoch = (start - pandas.Timestamp('1970-01-01')) // pandas.Timedelta('1s')
        stop_epoch = (stop - pandas.Timestamp('1970-01-01')) // pandas.Timedelta('1s')
        output = f'{filename}_{start_str}_{stop_str}{file_extension}'
        try:
            extract_from_file(input_name, output, start_epoch, stop_epoch)
        except NoData:
            print(f'No data in interval {start_str} - {stop_str}')
            continue


def parse_int(string, possible_values):
    try:
        val = int(string)
    except ValueError:
        sys.exit(f'Expected an integer, got "{string}"')
    if val not in possible_values:
        sys.exit(f'Expected an integer in {possible_values}, got "{val}"')
    return val

if __name__ == '__main__':
    t = time.time()
    if len(sys.argv) != 4:
        sys.exit(f'Syntax: {sys.argv[0]} <input HDF5 file> <first year> <period (in months)>')
    input_file = sys.argv[1]
    first_year = parse_int(sys.argv[2], range(1970, 2030))
    period = parse_int(sys.argv[3], {1, 2, 3, 4, 6, 12})
    extract_all(input_file, first_year, period)
    t = time.time() - t
    print(f'Terminated to process {input_file} in {t/60:.0f} minutes')
