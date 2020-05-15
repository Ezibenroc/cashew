import unittest
import tempfile
from random import random, randint, shuffle
import os
import pandas
import numpy
from numpy import dtype
from numpy.testing import assert_equal, assert_almost_equal, assert_raises
from io import StringIO
from cashew.archive_extraction import read_performance, read_database, write_database
import cashew.non_regression_tests as nrt


model_csv = '''
function,m,n,k,timestamp,duration,core,node,cluster,jobid,cpu,start_date,index
dgemm,10182,6081, 140,237.157994,5.425146e-01,29,1,dahu,1870094,1,2019-06-14,0
dgemm, 1441, 338,2072,329.063663,7.558411e-02,10,1,dahu,1870094,0,2019-06-14,0
dgemm, 3908,3279, 547,213.282718,4.462024e-01,23,1,dahu,1870094,1,2019-06-14,0
'''

def get_df(csv):
    df = pandas.read_csv(StringIO(csv))
    df.columns = df.columns.str.strip()
    return df

class ArchiveExtractionTest(unittest.TestCase):
    def check_dataframe_equality(self, df1, df2):
        self.assertEqual(len(df1), len(df2))
        self.assertEqual(set(df1.columns), set(df2.columns))
        # We allow the dataframes to have a different order
        df1 = df1.sort_values(by=list(sorted(df1.columns)))
        df2 = df2.sort_values(by=list(sorted(df2.columns)))
        for (_, row1), (_, row2) in zip(df1.iterrows(), df2.iterrows()):
            for col in df1.columns:
                self.assertEqual(df1[col].dtype, df2[col].dtype, msg='column %s' % col)
                if df1[col].dtype == dtype('float'):  # or df2[col].dtype == dtype('float'):
                    self.assertAlmostEqual(row1[col], row2[col], msg='column %s' % col)
                else:
                    self.assertEqual(row1[col], row2[col], msg='column %s' % col)

    def test_read_performance(self):
        df = read_performance('test_data/test_data.zip')
        expected = pandas.read_csv('test_data/test_data_performance.csv')
        self.check_dataframe_equality(expected, df)

    def check_read_write_database(self, expected):
        # Writing in one operation
        with tempfile.TemporaryDirectory() as tmpdir:
            database_name = os.path.join(tmpdir, 'database.db')
            write_database(expected, database_name)
            df = read_database(database_name)
            self.check_dataframe_equality(expected, df)
#        # Writing in two operations
#        with tempfile.TemporaryDirectory() as tmpdir:
#            database_name = os.path.join(tmpdir, 'database.db')
#            idx = len(expected)//2
#            write_database(expected.iloc[:idx], database_name)
#            write_database(expected.iloc[idx:], database_name)
#            df = read_database(database_name)
#            self.check_dataframe_equality(expected, df)

    def test_read_write_simple_database(self):
        self.check_read_write_database(get_df(model_csv))

    @staticmethod
    def generate_test_dataframe():
        rows = []
        jobid = 4242
        func = 'dgemm'
        cluster = 'dahu'
        for node in range(1, 33):
            for core in range(32):
                cpu = core % 2
                rows.append(dict(
                        function=func, m=randint(1, 1000), n=randint(1, 1000), k=randint(1, 1000),
                        timestamp=random(), duration=random(),
                        node=node, core=core, cluster=cluster, cpu=cpu,
                        jobid=jobid,
                    ))
        shuffle(rows)
        result = pandas.DataFrame(rows)
        assert len(result) > 50
        return result

    def test_read_write_complex_database(self):
        self.check_read_write_database(self.generate_test_dataframe())


class NonRegressionTest(unittest.TestCase):
    @staticmethod
    def get_changelog():
        df = get_df('''
        date,cluster,node,type,description
        2019-04-13,all,all,protocol,randomisation
        2019-08-15,dahu,13/14/15/16,G5K,cooling issue
        2019-10-18,all,all,protocol,randomisation again
        ''')
        df['date'] = pandas.to_datetime(df['date'])
        return df

    @staticmethod
    def get_dataframe():
        rows = ['timestamp,cluster,node,cpu,my_col,my_id']
        for i in range(1, 31):
            rows.append(f'''
            2019-07-{i:02d},dahu,10,0,100,A
            2019-07-{i:02d},dahu,10,1,200,B
            2019-07-{i:02d},dahu,14,0,300,C
            2019-07-{i:02d},dahu,14,1,400,D
            ''')
        for i in range(1, 31):
            rows.append(f'''
            2019-09-{i:02d},dahu,10,0,100,A
            2019-09-{i:02d},dahu,10,1,200,B
            2019-09-{i:02d},dahu,14,0,301,E
            2019-09-{i:02d},dahu,14,1,401,F
            ''')
        for i in range(1, 31):
            rows.append(f'''
            2019-12-{i:02d},dahu,10,0,101,G
            2019-12-{i:02d},dahu,10,1,201,H
            2019-12-{i:02d},dahu,14,0,302,I
            2019-12-{i:02d},dahu,14,1,402,J
            ''')
        rows = '\n'.join(rows)
        df = get_df(rows)
        df['timestamp'] = pandas.to_datetime(df['timestamp'])
        return df

    @staticmethod
    def get_dataframe_simple(cluster='dahu', node=42, cpu=0, colname='my_col', start_time=1580000000, N=100,
                             mu=12, sigma=4, seed=42):
        numpy.random.seed(seed)
        rows = []
        for i in range(start_time, start_time+N*10, 10):
            rows.append({
                'timestamp': pandas.to_datetime(i, unit='s'),
                'cluster': cluster,
                'node': node,
                'cpu': cpu,
                colname: numpy.random.normal(mu, sigma),
            })
        return pandas.DataFrame(rows)

    def test_mu_sigma(self):
        NA = float('NaN')
        nmin=8
        keep=3
        changelog = self.get_changelog()
        df = self.get_dataframe()
        marked=nrt.mark_weird(df, select_func=lambda x: nrt.select_after_changelog(x, changelog, nmin=nmin, keep=keep),
                naive=False, confidence=0.95, col="my_col")
        for key in df['my_id'].unique():
            tmp = marked[marked['my_id'] == key]
            avg = float(list(tmp['my_col'])[0])
            count = len(tmp)
            if key in ['A', 'B', 'C', 'D']:
                keep_prefix = False
                expected = [NA]*nmin + [avg]*(count-nmin)
            else:
                keep_prefix = True
                expected = [avg-1]*keep + [NA]*(nmin-keep) + [avg]*(count-nmin)
            real = list(tmp['mu'])
            assert_equal(real[keep:], expected[keep:])
            assert_almost_equal(real[:keep], expected[:keep], decimal=1)
            expected_sigma = [0*mu for mu in expected]
            real_sigma = list(tmp['sigma'])
            assert_equal(real_sigma[keep:], expected_sigma[keep:])
            assert_almost_equal(real_sigma[:keep], expected_sigma[:keep], decimal=0)
            if keep_prefix:
                assert_raises(AssertionError, assert_equal, real_sigma[:keep], expected_sigma[:keep])

    def test_simple_mu_sigma(self):
        nmin=8
        keep=3
        changelog = self.get_changelog()
        df = self.get_dataframe_simple(N=100)
        marked=nrt.mark_weird(df, select_func=lambda x: nrt.select_after_changelog(x, changelog, nmin=nmin, keep=keep),
                naive=False, confidence=0.95, col="my_col")
        expected_mu = list(marked['my_col'].expanding(nmin).mean().shift(1))
        expected_sigma = list(marked['my_col'].expanding(nmin).std().shift(1))
        expected_nbobs = list(marked['my_col'].expanding(nmin).count().shift(1))
        assert_almost_equal(list(marked['mu']), expected_mu)
        assert_equal(list(marked['nb_obs'])[nmin:], expected_nbobs[nmin:])

if __name__ == "__main__":
    unittest.main()
