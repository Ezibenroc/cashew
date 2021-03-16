import argparse
import time
import sys
from .archive_extraction import read_performance, read_monitoring, write_database
from .linear_regression import update_regression
from .notebook_generation import main as __main_notebook
from .version import __version__, __git_version__
from .logger import logger


def main_extract(args):
    parser = argparse.ArgumentParser(description='Archive extraction.')
    parser.add_argument('archive_name', type=str, help='Peanut archive file (input)')
    parser.add_argument('information', choices=['performance', 'monitoring'],
                        help='Which information to extract from the archive')
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
    if args.information == 'performance':
        df = read_performance(args.archive_name)
    elif args.information == 'monitoring':
        df = read_monitoring(args.archive_name)
    else:
        assert False
    write_database(df, args.database_name, **db_args)
    stop = time.time()
    logger.info('Processed archive %s containing %d rows in %.02f seconds' % (args.archive_name, len(df), stop-start))


def main_stats(args):
    parser = argparse.ArgumentParser(description='Statistics computation.')
    parser.add_argument('database_name', type=str, help='Database where are stored all the raw results (input)')
    parser.add_argument('output_name', type=str, help='Output file to store the statistics')
    parser.add_argument('--conditions', type=str, nargs='*', help='HDF conditions to filter the database.',
                        default=[])
    args = parser.parse_args(args)
    update_regression(args.database_name, args.output_name, conditions=args.conditions)


def main_notebook(args):
    possible_clusters = [
            'chetemi',
            'dahu',
            'yeti',
            'troll',
            'paravance',
            'parasilo',
            'grisou',
            'gros',
            'ecotype',
            'pyxis',
            'chiclet',
            'grvingt',
    ]
    parser = argparse.ArgumentParser(description='Generation of non-regression notebooks.')
    parser.add_argument('clusters', help='Name of the clusters to test', nargs='+',
            choices = possible_clusters + ['all'])
    parser.add_argument('--output', help='Directory to store the resulting files.',
                        type=str, default='/tmp')
    args = parser.parse_args(args)
    if 'all' in args.clusters:
        clusters = possible_clusters
    else:
        clusters = args.clusters
    __main_notebook(args.output, clusters)


def main():
    parser = argparse.ArgumentParser(description='Cashew, the peanut extractor')
    parser.add_argument('command', choices=['extract', 'stats', 'test'], help='Operation to perform.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument('--git-version', action='version',
                        version='%(prog)s {version}'.format(version=__git_version__))
    args, command_args = parser.parse_known_args()
    main_funcs = {
        'extract': main_extract,
        'stats': main_stats,
        'test': main_notebook,
    }
    try:
        main_funcs[args.command](command_args)
    except Exception as e:
        logger.error(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
