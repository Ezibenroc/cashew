import time
import sys
import asyncio
import os
from . import non_regression_tests as nrt


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
    await run(f'jupyter nbconvert {filename} --to html --output {dstname}')
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
        'mean_gflops',
        'mean_frequency',
        'mean_power_cpu',
        'mean_temperature',
        'intercept', 'mnk', 'mn', 'mk', 'nk', 'm', 'n', 'k',
        'intercept_residual', 'mnk_residual', 'mn_residual', 'mk_residual', 'nk_residual',
        'm_residual', 'n_residual', 'k_residual',
    ]
    src_notebook = os.path.join(output_dir, 'src.ipynb')
    with open(src_notebook, 'w') as f:
        f.write(notebook_str)
    # Now, let's download the CSV files, so the notebooks will not have to download them N times
    files = set(nrt.DATA_FILES.values())
    nrt.get(nrt.DEFAULT_CHANGELOG_URL)
    for f in files:
        nrt.get(nrt.DEFAULT_CSV_URL_PREFIX + f)
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
    "# Testing performance non-regression on Grid'5000 clusters\n",
    "\n",
    "Regular measures are made on the 363 nodes of 8 Grid'5000 clusters to keep track of their evolution. Three main metrics are collected: the average CPU performance (in Gflop/s), the average CPU frequency (in GHz) and the average CPU temperature (in °C)."
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
    "cluster = 'yeti'\n",
    "factor = 'mean_gflops'\n",
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
    "import io\n",
    "import plotnine\n",
    "plotnine.options.figure_size = 10, 7.5\n",
    "plotnine.options.dpi = 100\n",
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
    "csv_url = nrt.DEFAULT_CSV_URL_PREFIX + nrt.DATA_FILES[factor]\n",
    "df = nrt.format(nrt.get(csv_url))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "changelog = nrt.format_changelog(nrt.get(nrt.DEFAULT_CHANGELOG_URL))"
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
    "df = nrt.filter_na(df, factor)"
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
    "marked=nrt.mark_weird(df, changelog, nmin=10, keep=5, window=5, naive=False, confidence=confidence, col=factor)\n",
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
    "print(nrt.plot_overview_raw_data(marked, changelog))\n",
    "plotnine.options.figure_size = old_sizes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Anomalies\n",
    "\n",
    "The goal of the following cells is to detect the eventual anomalies for the considered metric (performance, frequency or temperature).\n",
    "\n",
    "Suppose that we have made 20 different experiments with a given CPU on a given node and measured its average temperature each time. We therefore have a list of 20 values. We can now compute:\n",
    "- $\\mu$ the sample mean of the 20 measures\n",
    "- $\\sigma$ the sample standard deviation of the 20 measures\n",
    "\n",
    "For instance, we may have $\\mu \\approx 64.7°C$ and $\\sigma \\approx 3.2°C$.\n",
    "\n",
    "Now, suppose that we perform a new experiment. This time, this CPU has an average temperature of $70°C$. This new temperature measure is higher than the mean of the 20 previous ones, but was it *significantly* too high? What was the probability of having a temperature at least as high if nothing changed on the CPU?\n",
    "\n",
    "In the evolution plots, we show the observed values with a prediction region $\\mu \\pm \\alpha\\times\\sigma$, where the factor $\\alpha$ is defined for a given confidence. With a conficence of 99.99%, if nothing has changed on the CPU, then 99.99% of the measures will fall in the prediction region. In other words, if a measure fall *outside* of this region, then there is probably something unusual that happened on this CPU at this time. The factor $\\alpha$ is computed using the [quantile function](https://en.wikipedia.org/wiki/Quantile_function) of either the [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) or the [F distribution](https://en.wikipedia.org/wiki/F-distribution).\n",
    "\n",
    "Back to our example, if we use the normal distribution, with a 99.99% confidence $\\alpha \\approx 3.89$ and the associated prediction region is $[52.3°C, 77.1°C]$. Our latest observation of $70°C$ falls in this region, so we consider that there is nothing unusual here.\n",
    "\n",
    "In the overview plots, the question is the other way around. We estimate what was the probability to observe a value as high (or as low) given the prior knowledge we had ($\\mu$ and $\\sigma$). First, we compute this probability (also called *likelihood*) using the [cumulative distribution function](https://en.wikipedia.org/wiki/Cumulative_distribution_function) of either the normal distribution or the F distribution. This probability can be very low, so for an easier visualization we take its logarithm. This new value, called *log-likelihood*, is always negative. For a better visualization, we then give it a sign (positive if the new observation is higher than the mean, negative otherwise). We also bound it to reasonable values to not distort too much the color scale.\n",
    "\n",
    "Back to our example, if we use the normal distribution, the probability to observe a value at least as high as $70°C$ was $L \\approx 0.049$. The log-likelihood is thus $LL \\approx -3.02$. Finally, the new observation was higher than the mean, so we give it a positive sign: the final value is $3.02$."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Brutal anomalies"
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
    "print(nrt.plot_overview(marked, changelog, confidence=confidence, discretize=True))\n",
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
    "node_limit = None if factor.startswith('mean') else 1\n",
    "nrt.plot_evolution_cluster(marked, changelog=changelog, node_limit=node_limit)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Long-term anomalies (window of 5 jobs)"
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
    "print(nrt.plot_overview_windowed(marked, changelog, confidence=confidence, discretize=True))\n",
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
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "node_limit = None if factor.startswith('mean') else 1\n",
    "nrt.plot_evolution_cluster_windowed(marked, changelog=changelog, node_limit=node_limit)"
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
