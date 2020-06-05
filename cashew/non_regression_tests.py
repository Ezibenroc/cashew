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


def _compute_mu_sigma(df, changelog, col, nmin, keep, window):
    '''
    For each (node, cpu) pair of the given dataframe, this function computes various summary values (such as mean and
    standard deviation) in between two changes of the given changelog.
    If less than `keep` measures have been made since the last change, then the summary values are made equal to the
    ones from the measure made just before this change.
    If more than `keep` but less than `nmin` measures have been made since the last change, then the summary values are
    not defined (NaN).

    **ASSUMPTION**: for a given node and a given CPU, the measures are sorted in increasing timestamp.
    '''
    cluster = select_unique(df, 'cluster')
    for node in df['node'].unique():
        local_changes = list(filter_changelog(changelog, cluster, node).sort_values(by='date')['date'])
        local_changes = [pandas.to_datetime('2000-01-01')] + local_changes + [pandas.to_datetime('2050-01-01')]
        intervals = list(zip(local_changes[:-1], local_changes[1:]))
        for cpu in df['cpu'].unique():
            cpu_mask = (df['node'] == node) & (df['cpu'] == cpu)
            assert df[cpu_mask]['timestamp'].is_monotonic_increasing
            for min_date, max_date in reversed(intervals):
                mask = (cpu_mask &
                       (df['timestamp'] >= min_date) & (df['timestamp'] < max_date))
                if keep > 0:
                    next_measures = (cpu_mask & (df['timestamp'] >= max_date))
                    next_measures = next_measures[next_measures].head(n=keep).index
                    mask.iloc[next_measures] = True
                local_df = df[mask]
                df.loc[mask, 'rolling_avg'] = local_df[col].rolling(window=window).mean()
                values = {
                    'mu'    : local_df[col].expanding(nmin).mean(),
                    'sigma' : local_df[col].expanding(nmin).std(),
                    'nb_obs': local_df[col].expanding(nmin).count(),
                    '__nb__': local_df[col].expanding().count(),
                }
                for key, series in values.items():
                    df.loc[mask, key] = series.shift(1)
                    df.loc[mask, f'{key}_current'] = series
                    df.loc[mask, f'{key}_old'] = series.shift(window)
                df.drop(['__nb__', '__nb___old', '__nb___current'], axis=1, inplace=True)


def _mark_weird(df, confidence, naive, window, col):
    '''
    Assume that the function _compute_mu_sigma has been called previously.
    '''
    df['standard_score'] = (df[col] - df['mu'])/df['sigma']
    if naive:
        one_side_conf = 1-(1-confidence)/2
        factor = stats.norm.ppf(one_side_conf)
        df['likelihood'] = 1-stats.norm.cdf(df['standard_score'].abs())
    else:
        base_factor = stats.f.ppf(confidence, 1, df['nb_obs']-1)
        factor = (base_factor*(df['nb_obs']+1)/df['nb_obs'])**(1/2)
        factor_windowed = (base_factor*(df['nb_obs']+window)/(df['nb_obs']*window))**(1/2)
        df['likelihood'] = 1-stats.f.cdf(df['standard_score']**2, 1, df['nb_obs']-1)
        score          = df['standard_score']**2 * (df['nb_obs'])/(df['nb_obs']+1)
        score_windowed = ((df['rolling_avg']-df['mu_old'])/df['sigma_old'])**2 * (df['nb_obs']*window)/(df['nb_obs']+window)
        df['likelihood'] = 1-stats.f.cdf(score, 1, df['nb_obs']-1)
        df['windowed_likelihood'] = 1-stats.f.cdf(score_windowed, 1, df['nb_obs']-1)
        df['windowed_log_likelihood'] = numpy.log(df['windowed_likelihood'])
    df['log_likelihood'] = numpy.log(df['likelihood'])
    # weirdness of 0 if positive log-likelihood, else sign(x-mu)*abs(log-likelihood)
    df['weirdness'] = df['log_likelihood']
    df.loc[df['log_likelihood'] >= 0, 'weirdness'] = 0
    df.loc[df['log_likelihood'] < 0, 'weirdness'] = numpy.sign(df[col]-df['mu'])*abs(df['log_likelihood'])
    df['low_bound']  = df['mu'] - df['sigma']*factor
    df['high_bound'] = df['mu'] + df['sigma']*factor
    df['weird_pos'] = df[col] - df['mu'] > factor*df['sigma']
    df['weird_neg'] = df[col] - df['mu'] < -factor*df['sigma']
    df['weird'] = (df['weird_pos'] | df['weird_neg']).astype(str)
    df.loc[df['weird_pos'] == True, 'weird'] = 'positive'
    df.loc[df['weird_neg'] == True, 'weird'] = 'negative'
    df.loc[df['mu'].isna(), 'weird'] = 'NA'
    # Then, the same thing but windowed
    if not naive:  # no factor_windowed otherwise
        df['windowed_weirdness'] = df['windowed_log_likelihood']
        df.loc[df['windowed_log_likelihood'] >= 0, 'windowed_weirdness'] = 0
        df.loc[df['windowed_log_likelihood'] < 0, 'windowed_weirdness'] = numpy.sign(df['rolling_avg']-df['mu_old'])*abs(df['windowed_log_likelihood'])
        df['windowed_low_bound']  = df['mu_old'] - df['sigma_old']*factor_windowed
        df['windowed_high_bound'] = df['mu_old'] + df['sigma_old']*factor_windowed
        df['windowed_weird_pos'] = df['rolling_avg'] - df['mu_old'] > factor_windowed*df['sigma_old']
        df['windowed_weird_neg'] = df['rolling_avg'] - df['mu_old'] < -factor_windowed*df['sigma_old']
        df['windowed_weird'] = (df['windowed_weird_pos'] | df['windowed_weird_neg']).astype(str)
        df.loc[df['windowed_weird_pos'] == True, 'windowed_weird'] = 'positive'
        df.loc[df['windowed_weird_neg'] == True, 'windowed_weird'] = 'negative'
        df.loc[df['mu_old'].isna(), 'windowed_weird'] = 'NA'
    return df


def mark_weird(df, changelog, confidence=0.95, naive=False, col='avg_gflops', nmin=8, keep=3, window=5):
    '''
    Mark the points of the given columns that are out of the prediction region of given confidence.
    The confidence should be a number between 0 and 1 (e.g. 0.95 for 95% confidence).
    If naive is True, then it assumes that the sample variane is exactly equal to the true variance, which results in a
    tighter prediction region.
    '''
    df = df.reset_index(drop=True).copy()
    _compute_mu_sigma(df, changelog, col=col, nmin=nmin, keep=keep, window=window)
    _mark_weird(df, confidence=confidence, naive=naive, col=col, window=window)
    df.window_size = window
    df.interest_col = col
    return df


def get_date_breaks(df):
    nb_days = (df['timestamp'].max() - df['timestamp'].min()).days
    interval = math.floor(nb_days / 3 / 30)
    if interval > 0:
        interval = f'{interval} months'
    else:
        interval = '2 weeks'
    return interval


def plot_evolution_node(df, col, low_col, high_col, weird_col):
    return ggplot(df) +\
            aes(x='timestamp', y=col) +\
            geom_line() +\
            geom_point(aes(fill=weird_col), size=1.5, stroke=0) +\
            geom_point(df[df[weird_col].isin({'positive', 'negative'})], aes(fill=weird_col), size=3, stroke=0) +\
            scale_fill_manual({
                'NA': '#AAAAAA',
                'positive': '#FF0000',
                'negative': '#0000FF',
                'False': '#00FF00'}) +\
            theme_bw() +\
            geom_ribbon(aes(ymin=low_col, ymax=high_col), color='grey', alpha=0.2) +\
            facet_wrap('cpu', labeller='label_both') +\
            scale_x_datetime(breaks=date_breaks(get_date_breaks(df)))


def _generic_plot_evolution(df, col, low_col, high_col, weird_col, changelog=None):
    mid = df[col].median()
    w = 0.2
    min_f = min(mid*(1-w), df['low_bound'].min())
    max_f = max(mid*(1+w), df['high_bound'].max())
    cluster = select_unique(df, 'cluster')
    for node in sorted(df['node'].unique()):
        print(f'{cluster}-{node}')
        plot = plot_evolution_node(df[df['node'] == node], col, low_col, high_col, weird_col) +\
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
            plot += geom_label(data=log[log['type'] == 'G5K'], mapping=aes(label='description', x='date', color='type'), y=max_f, size=8)
            plot += geom_label(data=log[log['type'] != 'G5K'], mapping=aes(label='description', x='date', color='type'), y=min_f, size=8)
        print(plot)


def plot_evolution_cluster(df, changelog=None):
    _generic_plot_evolution(df, df.interest_col, low_col='low_bound', high_col='high_bound', weird_col='weird', changelog=changelog)


def plot_evolution_cluster_windowed(df, changelog=None):
    _generic_plot_evolution(df, col='rolling_avg', low_col='windowed_low_bound', high_col='windowed_high_bound',
            weird_col='windowed_weird', changelog=changelog)


def _generic_overview(df, changelog, col, weird_col, grey_after_reset=True):
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
        aes(x='timestamp', y='node_cpu') +\
        geom_point(df[df[weird_col] == 'NA'], *[aes(fill=col) if not grey_after_reset else None],  **{**points_args, **({'fill': '#AAAAAA'} if grey_after_reset else {})}) +\
        geom_point(df[df[weird_col] == 'False'], aes(fill=col), **points_args) +\
        geom_point(df[~df[weird_col].isin({'NA', 'False'})], aes(fill=col), **points_args) +\
        geom_vline(global_changes, aes(xintercept='date', color='type'), size=1) +\
        scale_color_manual({
            'protocol': '#888888',
            'G5K': '#DD9500'},
            guide=False) +\
        theme_bw() +\
        scale_x_datetime(breaks=date_breaks(get_date_breaks(df))) +\
        ylab('Node:CPU') +\
        ggtitle(f'Overview of the cluster {cluster}')
    if len(local_changes) > 0:
        plot += geom_segment(local_changes, aes(x='date', xend='date', y='ymin', yend='ymax', color='type'),
                    position=position_nudge(y=0.5), size=1)
    return plot


class Color:
    '''
    The code from this class comes mainly from https://bsou.io/posts/color-gradients-with-python
    '''
    @staticmethod
    def hex_to_RGB(hex):
        ''' "#FFFFFF" -> [255,255,255] '''
        return [int(hex[i:i+2], 16) for i in range(1,6,2)]

    @staticmethod
    def RGB_to_hex(RGB):
        ''' [255,255,255] -> "#FFFFFF" '''
        # Components need to be integers for hex to make sense
        RGB = [int(x) for x in RGB]
        return "#"+"".join(["0{0:x}".format(v) if v < 16 else
                  "{0:x}".format(v) for v in RGB])

    @classmethod
    def linear_gradient(cls, start_hex, finish_hex="#FFFFFF", n=10):
        ''' returns a gradient list of (n) colors between
          two hex colors. start_hex and finish_hex
          should be the full six-digit color string,
          inlcuding the number sign ("#FFFFFF") '''
        # Starting and ending colors in RGB form
        s = cls.hex_to_RGB(start_hex)
        f = cls.hex_to_RGB(finish_hex)
        # Initilize a list of the output colors with the starting color
        RGB_list = [s]
        # Calcuate a color at each evenly spaced value of t from 1 to n
        for t in range(1, n):
            # Interpolate RGB vector for color at the current value of t
            curr_vector = [
              int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
              for j in range(3)
            ]
            # Add it to our list of output colors
            RGB_list.append(curr_vector)
        return [cls.RGB_to_hex(rgb) for rgb in RGB_list]


def _generic_overview_weirdness(df, changelog, confidence, weirdness_col, weird_col, likelihood_col, discretize):
    def round_to_1(x):
        return round(x, -int(numpy.floor(numpy.log10(abs(x)))))
    df = df.copy()
    if discretize:
        base_prob = 1-confidence
        probabilities = [0] + [round_to_1(base_prob*i) for i in [1, 10, 100, 1000]] + [1]
        probabilities = [p for p in probabilities if 0 <= p <= 1]
        prob_intervals = list(zip(probabilities[:-1], probabilities[1:]))
        prob_intervals = [(min_p, max_p) for (min_p, max_p) in prob_intervals if min_p != max_p]
        prob_str = [f'{min_p*100}% - {max_p*100}%' for min_p, max_p in prob_intervals]
        df['probability_str'] = prob_str[-1]
        for (min_p, max_p), proba_str in zip(prob_intervals[:-1], prob_str[:-1]):
            df.loc[(df[weirdness_col] > 0) & (df[likelihood_col] >= min_p) & (df[likelihood_col] < max_p),
                    'probability_str'] = f'[+] {proba_str}'
            df.loc[(df[weirdness_col] < 0) & (df[likelihood_col] >= min_p) & (df[likelihood_col] < max_p),
                    'probability_str'] = f'[-] {proba_str}'
        final_prob_str =  [f'[+] {p}' for p in prob_str[:-1]]
        final_prob_str += [prob_str[-1]]
        final_prob_str += [f'[-] {p}' for p in reversed(prob_str[:-1])]
        df['probability_str'] = pandas.Categorical(df['probability_str'], categories=final_prob_str, ordered=True)
        colors =  Color.linear_gradient('#FF0000', '#00FF00', n=len(prob_str))
        colors += Color.linear_gradient('#00FF00', '#0000FF', n=len(prob_str))[1:]
        plot = _generic_overview(df, changelog, 'probability_str', weird_col) +\
            scale_fill_manual(values = colors) +\
            labs(fill='Probability')
    else:
        weirdness_limit = abs(numpy.log(1-stats.f.cdf(stats.f.ppf(confidence, 1, 20), 1, 20)))
        print(f'Cutting the log-likelihood at {weirdness_limit:.2f} (due to the {confidence*100}% confidence)')
        df['bounded_weirdness'] = df[weirdness_col]
        df.loc[df[weirdness_col] > weirdness_limit, 'bounded_weirdness'] = weirdness_limit
        df.loc[df[weirdness_col] < -weirdness_limit, 'bounded_weirdness'] = -weirdness_limit
        plot = _generic_overview(df, changelog, 'bounded_weirdness', weird_col) +\
            scale_fill_gradient2(low='#0000FF', mid='#00FF00', high='#FF0000', limits=[-weirdness_limit, weirdness_limit]) +\
            labs(fill='Anomaly')
    return plot


def plot_overview(df, changelog, confidence=0.95, discretize=False):
    return _generic_overview_weirdness(df, changelog, confidence, 'weirdness', 'weird', 'likelihood', discretize)


def plot_overview_windowed(df, changelog, confidence=0.95, discretize=False):
    return _generic_overview_weirdness(df, changelog, confidence, 'windowed_weirdness', 'windowed_weird',
    'windowed_likelihood', discretize)


def plot_overview_raw_data(df, changelog):
    plot = _generic_overview(df, changelog, df.interest_col, 'weird', grey_after_reset=False) +\
        scale_fill_gradient2(low='#800080', mid='#EEEEEE', high='#FFA500', midpoint=df[df.interest_col].mean())
    return plot


def plot_overview_raw_data_windowed(df, changelog):
    plot = _generic_overview(df.dropna(subset=['rolling_avg']), changelog, 'rolling_avg', 'windowed_weird', grey_after_reset=False) +\
        scale_fill_gradient2(low='#800080', mid='#EEEEEE', high='#FFA500', midpoint=df['rolling_avg'].mean())
    return plot
