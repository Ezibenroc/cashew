import argparse
import time
from .archive_extraction import read_archive, write_database
from .version import __version__, __git_version__


def main():
    parser = argparse.ArgumentParser(description='Cashew, the peanut extractor')
    parser.add_argument('archive_name', type=str, help='Peanut archive file (input)')
    parser.add_argument('csv_name', type=str, help='Name of the CSV file of interest in the archive')
    parser.add_argument('database_name', type=str, help='Database where should be stored all the results (output)')
    parser.add_argument('table_name', type=str, help='Name of the table in the database')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument('--git-version', action='version',
                        version='%(prog)s {version}'.format(version=__git_version__))
    args = parser.parse_args()
    start = time.time()
    df = read_archive(args.archive_name, args.csv_name)
    write_database(df, args.database_name, args.table_name, how='sql', compress=False)
    stop = time.time()
    print('Processed archive %s containing %d rows in %.02f seconds' % (args.archive_name, len(df), stop-start))


if __name__ == '__main__':
    main()
