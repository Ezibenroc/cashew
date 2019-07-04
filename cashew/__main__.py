import argparse
import time
import sys
from .archive_extraction import read_archive, write_database
from .linear_regression import update_regression
from .version import __version__, __git_version__
from .logger import logger


def main_extract(args):
    parser = argparse.ArgumentParser(description='Archive extraction.')
    parser.add_argument('archive_name', type=str, help='Peanut archive file (input)')
    parser.add_argument('csv_name', type=str, help='Name of the CSV file of interest in the archive')
    parser.add_argument('database_name', type=str, help='Database where should be stored all the results (output)')
    parser.add_argument('--compression', type=str, choices=['none', 'zlib', 'bzip2', 'lzo',  'blosc'],
                        help='Compression mode for the database', default='blosc')
    parser.add_argument('--compression_lvl', type=int, choices=range(1, 10),
                        help='Compression level for the database', default=0)
    parser.add_argument('--format', type=str, choices=['fixed', 'table'],
                        help='Data layout in the database', default='fixed')
    args = parser.parse_args(args)
    db_args = {}
    if args.compression_lvl > 0 and args.compression == 'none':
            parser.error('Specified a compression level whereas no compression was asked')
    if args.compression_lvl == 0 and args.compression != 'none':
        args.compression_lvl = 9
    if args.compression != 'none':
        db_args['complib'] = args.compression
        db_args['complevel'] = args.compression_lvl
    db_args['format'] = args.format
    if args.format == 'table':
        db_args['data_columns'] = ['function', 'cluster', 'node', 'jobid', 'start_time']
        db_args['append'] = True
    start = time.time()
    df = read_archive(args.archive_name, args.csv_name)
    write_database(df, args.database_name, **db_args)
    stop = time.time()
    logger.info('Processed archive %s containing %d rows in %.02f seconds' % (args.archive_name, len(df), stop-start))


def main_stats(args):
    parser = argparse.ArgumentParser(description='Statistics computation.')
    parser.add_argument('database_name', type=str, help='Database where are stored all the raw results (input)')
    parser.add_argument('output_name', type=str, help='Output file to store the statistics')
    args = parser.parse_args(args)
    update_regression(args.database_name, args.output_name)


def main():
    parser = argparse.ArgumentParser(description='Cashew, the peanut extractor')
    parser.add_argument('command', choices=['extract', 'stats'], help='Operation to perform.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument('--git-version', action='version',
                        version='%(prog)s {version}'.format(version=__git_version__))
    args, command_args = parser.parse_known_args()
    main_funcs = {
        'extract': main_extract,
        'stats': main_stats,
    }
    try:
        main_funcs[args.command](command_args)
    except Exception as e:
        logger.error(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
