import requests
import pandas
import io
import math
import logging
from scipy import stats
import numpy
import plotnine
plotnine.options.figure_size = (10, 7.5)
from plotnine import *
from mizani.breaks import date_breaks
from mizani.formatters import date_format

# Setting up a logger
logger = logging.getLogger('non_regression_tests')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def get(url):
    '''
    Download a CSV file at the specified URL and load it into a dataframe.
    '''
    data = requests.get(url)
    if data.status_code != 200:
        raise ValueError(f'Could not download the CSV file, got an error {data.status_code}')
    df = pandas.read_csv(io.BytesIO(data.content))
    logger.info(f'Downloaded a dataframe with {len(df)} rows and {len(df.columns)} columns')
    return df


def get_path(path):
    '''
    Load a CSV file from the specified path.
    '''
    df = pandas.read_csv(path)
    logger.info(f'Downloaded a dataframe with {len(df)} rows and {len(df.columns)} columns')
    return df


def format(df):
    '''
    Apply various formating on the given dataframe (e.g., datetime parsing).
    '''
    df['timestamp'] = pandas.to_datetime(df['start_time'], unit='s')
    return df


def format_changelog(df):
    df['date'] = pandas.to_datetime(df['date'])
    return df


def filter_changelog(df, cluster, node):
    result_rows = []
    for _, row in df.iterrows():
        if row['cluster'] == 'all' or cluster in row['cluster'].split('/'):
            if row['node'] == 'all' or str(node) in row['node'].split('/'):
                result_rows.append(row)
    if len(result_rows) > 0:
        result = pandas.DataFrame(result_rows)
    else:
        result = pandas.DataFrame(columns=df.columns)
    return result


def get_changes_from_changelog(df, cluster):
    global_changes = []
    local_changes = []
    for _, row in df.iterrows():
        if row['cluster'] == 'all' or cluster in row['cluster'].split('/'):
            event = (row['date'], row['type'])
            if row['node'] == 'all':
                global_changes.append(event)
            else:
                for node in row['node'].split('/'):
                    node = int(node)
                    local_changes.append(event + (node,))
    cols = 'date', 'type'
    return pandas.DataFrame(global_changes, columns=cols), pandas.DataFrame(local_changes, columns=cols+('node',))

def filter(df, **kwargs):
    '''
    Filter the dataframe according to the given named parameters.

    Example:
    filter(df, cluster='dahu')
    '''
    for key, value in kwargs.items():
        if key not in df.columns:
            raise ValueError(f'Cannot filter on {key}, missing column')
        df = df[df[key] == value]
    logger.info(f'Filtered the dataframe, there remains {len(df)} rows')
    return df


def filter_latest(df):
    '''
    Keep only the most recent run for each node of the dataframe.
    '''
    df = df.copy()
    all_nodes = [(cluster, node) for _, (cluster, node) in df[['cluster', 'node']].drop_duplicates().iterrows()]
    for cluster, node in all_nodes:
        mask = (df['cluster'] == cluster) & (df['node'] == node)
        last_run = df[mask]['start_time'].max()
        df.drop(df[mask & (df['start_time'] < last_run)].index, inplace=True)
    logger.info(f'Filtered the dataframe, there remains {len(df)} rows')
    return df


def select_unique(df, col):
    res = df[col].unique()
    assert len(res) == 1
    return res[0]


def plot_latest_distribution(df, col='avg_gflops'):
    min_f = df[col].min()
    max_f = df[col].max()
    cluster = select_unique(df, 'cluster')
    df = filter_latest(df)
    median = df[col].median()
    title = f'Distribution of the latest runs made on the cluster {cluster}\nMedian of {median:.2f}'
    return ggplot(df) +\
            aes(x=col) +\
            geom_histogram(binwidth=(max_f-min_f)/10, alpha=0.5) +\
            theme_bw() +\
            geom_vline(xintercept=median) +\
            expand_limits(x=(min_f, max_f)) +\
            ylab('Number of CPU') +\
            ggtitle(title)


def select_last_n(df, n=10):
    selection = df.tail(n=n)
    if len(df) < n:
        selection = pandas.DataFrame(columns=df.columns)
    return selection


def select_after_changelog(df, changelog, nmin=8, nmax=None, keep=0):
    '''
    Asusmption: df contains data for a single node of a single cluster.
    Return measures that have been made after the last event regarding this node. It returns at least nmin measures, and
    at most nmax. If nmax is not specified, it will return all of them.

    When keep is non-null, if the last measure(s) of the dataframe is/are the first one(s) done after a change, then
    this change is discarded so as to return a larger dataframe. The goal is to show in the plots the anomalies that
    allowed us to detect the changes.
    '''
    empty = pandas.DataFrame(columns=df.columns)
    if len(df) == 0:
        return empty
    cluster = select_unique(df, 'cluster')
    node = select_unique(df, 'node')
    changelog = filter_changelog(changelog, cluster, node)
    # We remove all the changes that will happen after the most recent event
    changelog = changelog[changelog['date'] <= df['timestamp'].max()]
    # We also remove the most recent event
    df = df[df['timestamp'] < df['timestamp'].max()]
    # Then, we remove all the events that have happened before the most recent change
    max_change = changelog['date'].max()
    if max_change != max_change:  # max_change is NaT (there was no change yet)
        max_change = pandas.to_datetime(0, unit='s')
    result = df[df['timestamp'] >= max_change]
    # Now, if there are too few events, it means we are right after the change, so we discard this change
    if keep > 0 and len(result) < keep:
        assert keep < nmin
        old_max = max_change
        max_change = changelog[changelog['date'] < max_change]['date'].max()
        if max_change != max_change:  # max_change is NaT (there was no change yet)
            max_change = pandas.to_datetime(0, unit='s')
        result = df[df['timestamp'] >= max_change]
    # Finally, we take the first nmax (if nmax is specified)
    if nmax is not None:
        result = result.sort_values(by='timestamp').head(n=nmax)
    if len(result) < nmin:
        return empty
    else:
        return result


def mark_weird(df, select_func=select_last_n, confidence=0.95, naive=False, col='avg_gflops'):
    '''
    Mark the points of the given columns that are out of the prediction region of given confidence.
    The confidence should be a number between 0 and 1 (e.g. 0.95 for 95% confidence).
    If naive is True, then it assumes that the sample variane is exactly equal to the true variance, which results in a
    tighter prediction region.
    '''
    df = df.copy()
    NAN = float('NaN')
    df['mu'] = NAN
    df['sigma'] = NAN
    df['nb_obs'] = NAN
    for i in range(0, len(df)):
        row = df.iloc[i]
        candidates = df[(df['node'] == row['node']) & (df['cpu'] == row['cpu']) & (df['timestamp'] <= row['timestamp'])]
        selected = select_func(candidates)#[col]
        selected = selected[col]
        df.loc[df.index[i], ('mu', 'sigma', 'nb_obs')] = selected.mean(), selected.std(), len(selected)
    df['standard_score'] = (df[col] - df['mu'])/df['sigma']
    if naive:
        one_side_conf = 1-(1-confidence)/2
        factor = stats.norm.ppf(one_side_conf)
        df['likelihood'] = stats.norm.pdf(df['standard_score'])
    else:
        factor = stats.f.ppf(confidence, 1, df['nb_obs']-1)*(df['nb_obs']+1)/df['nb_obs']
        factor = factor**(1/2)
        df['likelihood'] = stats.f.pdf(df['standard_score']**2, 1, df['nb_obs']-1)
    df['log_likelihood'] = numpy.log(df['likelihood'])
    # weirdness of 0 if positive log-likelihood, else sign(x-mu)*abs(log-likelihood)
    df['weirdness'] = df['log_likelihood']
    df.loc[df['log_likelihood'] >= 0, 'weirdness'] = 0
    df.loc[df['log_likelihood'] < 0, 'weirdness'] = numpy.sign(df[col]-df['mu'])*abs(df['log_likelihood'])
    df['low_bound']  = df['mu'] - df['sigma']*factor
    df['high_bound'] = df['mu'] + df['sigma']*factor
    df['weird_pos'] = df[col] - df['mu'] > factor*df['sigma']
    df['weird_neg'] = df[col] - df['mu'] < -factor*df['sigma']
    df['weird'] = df['weird_pos'] | df['weird_neg']
    df.loc[df['weird_pos'] == True, 'weird'] = 'positive'
    df.loc[df['weird_neg'] == True, 'weird'] = 'negative'
    df.loc[df['mu'].isna(), 'weird'] = 'NA'
    return df


def get_date_breaks(df):
    nb_days = (df['timestamp'].max() - df['timestamp'].min()).days
    interval = math.floor(nb_days / 3 / 30)
    if interval > 0:
        interval = f'{interval} months'
    else:
        interval = '2 weeks'
    return interval


def plot_evolution_node(df, col):
    return ggplot(df) +\
            aes(x='timestamp', y=col) +\
            geom_line() +\
            geom_point(aes(fill='weird'), size=1.5, stroke=0) +\
            geom_point(df[df.weird.isin({'positive', 'negative'})], aes(fill='weird'), size=3, stroke=0) +\
            scale_fill_manual({
                'NA': '#AAAAAA',
                'positive': '#FF0000',
                'negative': '#0000FF',
                False: '#00FF00'}) +\
            theme_bw() +\
            geom_ribbon(aes(ymin='low_bound', ymax='high_bound'), color='grey', alpha=0.2) +\
            facet_wrap('cpu', labeller='label_both') +\
            scale_x_datetime(breaks=date_breaks(get_date_breaks(df)))


def plot_evolution_cluster(df, col, changelog=None):
    mid = df[col].median()
    w = 0.2
    min_f = min(mid*(1-w), df['low_bound'].min())
    max_f = max(mid*(1+w), df['high_bound'].max())
    cluster = select_unique(df, 'cluster')
    for node in sorted(df['node'].unique()):
        print(f'{cluster}-{node}')
        plot = plot_evolution_node(df[df['node'] == node], col) +\
                ggtitle(f'Evolution of the node {cluster}-{node}') +\
                expand_limits(y=(min_f, max_f))
        if changelog is not None:
            log = filter_changelog(changelog[changelog['date'] >= df['timestamp'].min()], cluster, node)
            plot += geom_vline(log, aes(xintercept='date', color='type'), linetype='dashed')
            plot += scale_color_manual({
                'protocol': '#888888',
                'G5K': '#DD9500'},
                guide=False)
            log.loc[log['type'] == 'protocol', 'description'] = 'protocol'
            plot += geom_label(log[log['type'] == 'G5K'], aes(label='description', x='date', color='type'), y=max_f, size=8)
            plot += geom_label(log[log['type'] != 'G5K'], aes(label='description', x='date', color='type'), y=min_f, size=8)
        print(plot)


def _generic_overview(df, changelog, col, grey_after_reset=True):
    cluster = select_unique(df, 'cluster')
    df = df.copy()
    df['node_cpu'] = df['node'].astype(str) + ':' + df['cpu'].astype(str)
    node_cat = df[['node', 'cpu', 'node_cpu']].drop_duplicates().sort_values(by=['node', 'cpu'], ascending=False)['node_cpu']
    df['node_cpu'] = pandas.Categorical(df['node_cpu'], categories=node_cat, ordered=True)
    global_changes, local_changes = get_changes_from_changelog(changelog[changelog['date'] >= df['timestamp'].min()], cluster)
    local_changes['ymin'] = local_changes['node'].astype(str) + ':' + str(df['cpu'].min())
    local_changes['ymax'] = (local_changes['node']+1).astype(str) + ':' + str(df['cpu'].min())
    local_changes[col] = 42  # not used, but otherwise plotnine complains...
    points_args = {'stroke': 0, 'size': 3}
    plot = ggplot() +\
        aes(x='timestamp', y='node_cpu', fill=col) +\
        geom_point(df[df.weird == 'NA'], **{**points_args, **({'fill': '#AAAAAA'} if grey_after_reset else {})}) +\
        geom_point(df[df.weird == False], **points_args) +\
        geom_point(df[~df.weird.isin({'NA', False})], **points_args) +\
        geom_vline(global_changes, aes(xintercept='date', color='type'), size=1) +\
        geom_segment(local_changes, aes(x='date', xend='date', y='ymin', yend='ymax', color='type'),
                    position=position_nudge(y=0.5), size=1) +\
        scale_color_manual({
            'protocol': '#888888',
            'G5K': '#DD9500'},
            guide=False) +\
        theme_bw() +\
        scale_x_datetime(breaks=date_breaks(get_date_breaks(df))) +\
        ylab('Node:CPU') +\
        ggtitle(f'Overview of the cluster {cluster}')
    return plot


def plot_overview(df, changelog, confidence=0.95):
    weirdness_limit = abs(numpy.log(stats.f.pdf(stats.f.ppf(confidence, 1, 20), 1, 20)))
    print(f'Cutting the log-likelihood at {weirdness_limit:.2f} (due to the {confidence*100}% confidence)')
    df['bounded_weirdness'] = df['weirdness']
    df.loc[df['weirdness'] > weirdness_limit, 'bounded_weirdness'] = weirdness_limit
    df.loc[df['weirdness'] < -weirdness_limit, 'bounded_weirdness'] = -weirdness_limit
    plot = _generic_overview(df, changelog, 'bounded_weirdness') +\
        scale_fill_gradient2(low='#0000FF', mid='#00FF00', high='#FF0000', limits=[-weirdness_limit, weirdness_limit]) +\
        labs(fill='Anomaly')
    return plot


def plot_overview_raw_data(df, changelog, col):
    plot = _generic_overview(df, changelog, col, grey_after_reset=False) +\
        scale_fill_gradient2(low='#800080', mid='#EEEEEE', high='#FFA500', midpoint=df[col].mean())
    return plot
