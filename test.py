import unittest
import tempfile
from random import random, randint, shuffle
import os
import pandas
from numpy import dtype
from io import StringIO
from cashew.archive_extraction import read_archive, read_database, write_database


model_csv = '''
function,m,n,k,timestamp,duration,core,node,cluster,jobid,cpu,start_date,index
dgemm,10182,6081, 140,237.157994,5.425146e-01,29,1,dahu,1870094,1,2019-06-14,0
dgemm, 1441, 338,2072,329.063663,7.558411e-02,10,1,dahu,1870094,0,2019-06-14,0
dgemm, 3908,3279, 547,213.282718,4.462024e-01,23,1,dahu,1870094,1,2019-06-14,0
'''


class BasicTest(unittest.TestCase):
    @staticmethod
    def get_df(csv):
        return pandas.read_csv(StringIO(csv))

    def check_dataframe_equality(self, df1, df2):
        self.assertEqual(len(df1), len(df2))
        self.assertEqual(set(df1.columns), set(df2.columns))
        # We allow the dataframes to have a different order
        df1 = df1.sort_values(by=list(sorted(df1.columns)))
        df2 = df2.sort_values(by=list(sorted(df2.columns)))
        for (_, row1), (_, row2) in zip(df1.iterrows(), df2.iterrows()):
            for col in df1.columns:
                self.assertEqual(df1[col].dtype, df2[col].dtype)
                if df1[col].dtype == dtype('float'):  # or df2[col].dtype == dtype('float'):
                    self.assertAlmostEqual(row1[col], row2[col], msg='column %s' % col)
                else:
                    self.assertEqual(row1[col], row2[col], msg='column %s' % col)

    def test_read_archive(self):
        df = read_archive('test_data.zip', 'result.csv')
        expected = self.get_df(model_csv)
        self.check_dataframe_equality(expected, df)

    def check_read_write_database(self, expected):
        # Writing in one operation
        for compress in [True, False]:
            for how in ['sql', 'csv']:
                with tempfile.TemporaryDirectory() as tmpdir:
                    database_name = os.path.join(tmpdir, 'database.db')
                    table_name = 'my_table'
                    write_database(expected, database_name, table_name, compress=compress, how=how)
                    df = read_database(database_name, table_name, compress=compress, how=how)
                    self.check_dataframe_equality(expected, df)
                # Writing in two operations
                with tempfile.TemporaryDirectory() as tmpdir:
                    database_name = os.path.join(tmpdir, 'database.db')
                    table_name = 'my_table'
                    idx = len(expected)//2
                    write_database(expected.iloc[:idx], database_name, table_name, compress=compress, how=how)
                    write_database(expected.iloc[idx:], database_name, table_name, compress=compress, how=how)
                    df = read_database(database_name, table_name, compress=compress, how=how)
                    self.check_dataframe_equality(expected, df)

    def test_read_write_simple_database(self):
        self.check_read_write_database(self.get_df(model_csv))

    @staticmethod
    def generate_test_dataframe():
        rows = []
        for func in ['dgemm', 'mmegd', 'foobar']:
            for cluster in ['dahu', 'yeti', 'grvingt']:
                for node in [1, 18, 42]:
                    for core in [5, 9, 12]:
                        for jobid in [12345, 54321, 99999]:
                            cpu = core % 2
                            rows.append(dict(
                                    function=func, m=randint(1, 1000), n=randint(1, 1000), k=randint(1, 1000),
                                    timestamp=random(), duration=random(),
                                    node=node, core=core, cluster=cluster, cpu=cpu,
                                    jobid=jobid,
                                ))
        shuffle(rows)
        result = pandas.DataFrame(rows)
        assert len(result) > 200
        return result

    def test_read_write_complex_database(self):
        self.check_read_write_database(self.generate_test_dataframe())


if __name__ == "__main__":
    unittest.main()
