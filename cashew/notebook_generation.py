import time
import sys
import asyncio
import os


async def run(cmd):
    '''
    From https://docs.python.org/3/library/asyncio-subprocess.html
    '''
    print(cmd)
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()


async def run_notebook(src_notebook, dst_dir, cluster, parameter):
    dst_notebook = os.path.join(dst_dir, f'{cluster}_{parameter}.ipynb')
    await run(f'papermill {src_notebook} {dst_notebook} -p cluster {cluster} -p factor {parameter}')
    return dst_notebook


async def convert_notebook(filename):
    assert filename.endswith('.ipynb')
    dstname = filename[:-len('.ipynb')] + '.html'
    await run(f'jupyter nbconvert {filename} --output {dstname}')
    return dstname


async def process_notebook(src_notebook, dst_dir, cluster, parameter):
    tmp_file = await run_notebook(src_notebook, dst_dir, cluster, parameter)
    return await convert_notebook(tmp_file)


async def process_all(src_notebook, dst_dir, clusters, parameters):
    await asyncio.gather(*[
        process_notebook(src_notebook, dst_dir, clust, param)
        for clust in clusters
        for param in parameters
    ])


def main(output_dir, cluster_list):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    clusters = list(cluster_list)
    parameters = [
        'avg_gflops',
        'mean_frequency',
        'mean_temperature'
    ]
    src_notebook = os.path.join(output_dir, 'src.ipynb')
    with open(src_notebook, 'w') as f:
        f.write(notebook_str)
    t = time.time()
    asyncio.run(process_all(src_notebook, output_dir, clusters, parameters))
    t = time.time() - t
    print(f'Processed {len(clusters)} clusters for {len(parameters)} parameters in {t:.2f} seconds')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cluster_list = sys.argv[1:]
    else:
        cluster_list = None
    main(cluster_list)


notebook_str = r'''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Testing performance non-regression on Grid'5000 clusters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": [
     "parameters"
    ]
   },
   "outputs": [],
   "source": [
    "csv_url_prefix = 'https://gitlab.in2p3.fr/tom.cornebize/g5k_data_non_regression/raw/master/'\n",
    "changelog_url = 'https://gitlab.in2p3.fr/tom.cornebize/g5k_data_non_regression/raw/master/exp_changelog.csv'\n",
    "cluster = 'yeti'\n",
    "factor = 'avg_gflops'\n",
    "confidence = 0.9999"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "import requests\n",
    "import pandas\n",
    "import io"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_files = {\n",
    "    'avg_gflops': 'stats.csv',\n",
    "    'mean_temperature': 'stats_monitoring.csv',\n",
    "    'mean_frequency': 'stats_monitoring.csv',\n",
    "}\n",
    "csv_url = csv_url_prefix + all_files[factor]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from cashew import non_regression_tests as nrt\n",
    "import cashew\n",
    "print(cashew.__git_version__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "df = nrt.format(nrt.get(csv_url))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "changelog = nrt.format_changelog(nrt.get(changelog_url))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = nrt.filter(df, cluster=cluster)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "nrt.plot_latest_distribution(df, factor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "marked=nrt.mark_weird(df, changelog, nmin=8, keep=4, naive=False, confidence=confidence, col=factor)\n",
    "nb_weird = len(marked[marked.weird.isin({'positive', 'negative'})])\n",
    "nb_total = len(marked[marked.weird != 'NA'])\n",
    "print(f'{nb_weird/nb_total*100:.2f}% of measures are abnormal ({nb_weird}/{nb_total})')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "import plotnine\n",
    "nb_unique = len(marked[['node', 'cpu']].drop_duplicates())\n",
    "height = max(6, nb_unique/8)\n",
    "old_sizes = tuple(plotnine.options.figure_size)\n",
    "plotnine.options.figure_size = (10, height)\n",
    "print(nrt.plot_overview_raw_data(marked, changelog, factor))\n",
    "plotnine.options.figure_size = old_sizes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "plotnine.options.figure_size = (10, height)\n",
    "print(nrt.plot_overview(marked, changelog, confidence=confidence))\n",
    "plotnine.options.figure_size = old_sizes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "nrt.plot_evolution_cluster(marked, factor, changelog)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
