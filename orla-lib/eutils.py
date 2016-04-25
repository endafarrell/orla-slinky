import StringIO
import collections
import datetime
import gzip
import os
import subprocess
import csv
from contextlib import contextmanager
import multiprocessing
import time
import threading
import sys
import traceback
import codecs
import json
import pprint
import zlib
import base64

import psycopg2

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy


from steps import basedir


reload(sys)
# noinspection PyUnresolvedReferences
sys.setdefaultencoding('utf8')


def pipegzip(outf):
    proc = subprocess.Popen(["gzip", "-c", "-"], stdout=open(outf, 'w'), stdin=subprocess.PIPE)
    return proc.stdin


@contextmanager
def gzsubproc(outf):
    """wrap gzip subprocess in a context manager that will close and wait on exit"""
    proc = gzsubproc_enter(outf)
    yield gzsubproc_yield(proc)
    gzsubproc_exit(proc)


# gzsubproc components, exposed for use outside of a with clause

def gzsubproc_enter(outf):
    return subprocess.Popen(["gzip", "-c", "-"], stdout=open(outf, 'w'), stdin=subprocess.PIPE)


def gzsubproc_yield(proc):
    return proc.stdin


def gzsubproc_exit(proc):
    proc.stdin.close()
    proc.wait()


@contextmanager
def mppool(size=None):
    """wrap a pool in a context manager that waits for all tasks to finish
       (note that pool itself implements context manager, but on exit it terminates
       immeditaely, without waiting for work to finish)"""
    if size is not None:
        p = multiprocessing.Pool(size)
    else:
        p = multiprocessing.Pool()
    yield p
    p.close()
    p.join()


def fromgzip(inf):
    proc = subprocess.Popen(["zcat", inf], stdout=subprocess.PIPE)
    return proc.stdout


def connect(dsn):
    # # TODO - explain why with conda this host= is needed
    # if 'dbname' not in dsn:
    #     dsn = "dbname={} host=/var/run/postgresql/".format(dsn)
    # # TODO

    if 'dbname' not in dsn:
        dsn = "dbname={}".format(dsn)
    return psycopg2.connect(dsn)


def csvmap(filename):
    with open(basedir + "/data/" + filename) as inf:
        return dict(csv.reader(inf))


def pipetee(outf):
    """create a pipe that tees to a file and lets stdout and stderr contineu on, return [pipe, proc]"""
    proc = subprocess.Popen(["tee", outf], stdin=subprocess.PIPE)
    return proc.stdin, proc


def do_query(dbname, query, cur=None, explain=False):
    """ a method that wraps log prints around a query. """
    has_cur = True
    if cur is None:
        db = connect(dbname)
        cur = db.cursor()
        has_cur = False
    start = time.time()

    if type(query) is tuple and explain:
        raise Exception('Explain not possible for parameterized')

    if explain:
        # Does not work on ANALYZE. it breaks.
        q_temp = query.lower().strip(';').strip().strip('\t').strip('\n').strip(';')
        if q_temp.startswith('analyze'):
            explain = False
        if q_temp.startswith('drop'):
            explain = False
        if q_temp.startswith('create'):
            explain = False
        if ';' in q_temp:
            explain = False

        if explain:
            cur.execute('EXPLAIN %s' % query)
            explain = ''
            for row in cur:
                explain += '%s\n' % row
            print 'Queryplan:\n %s' % explain
            sys.stdout.flush()

    try:
        if type(query) is tuple:
            cur.executemany(query[0], query[1])
        else:
            cur.execute(query)
    except:
        traceback.print_exc(file=sys.stderr)
        raise Exception('This query is not working: \n %s' % (query[0] if type(query) is tuple else query))

    if type(query) is tuple:

        print """Parameterized query: \n %(query)s\n\tTime: %(time)f seconds\n\tEntries: %(status)s 
            """ % {
            "query": query[0],
            "time": time.time() - start,
            "status": len(query[1])}
    else:
        print """Query: \n %(query)s\n\tTime: %(time)f seconds\n\tStats: %(status)s 
            """ % {
            "query": cur.query,
            "time": time.time() - start,
            "status": cur.statusmessage}
    if not has_cur:
        # noinspection PyUnboundLocalVariable
        db.commit()
        db.close()

    return cur


def do_query_explain(dbname, query, cur=None):
    do_query(dbname, query, cur, True)


def def_query_method(dbname, query, cur=None):
    if cur:
        raise Exception("Not supported yet!")
    db = connect(dbname)
    cur = db.cursor()
    try:
        if type(query) is tuple:
            cur.executemany(query[0], query[1])
        else:
            cur.execute(query)
    except:
        traceback.print_exc(file=sys.stderr)
        raise Exception('This query is not working: \n %s' % (query[0] if type(query) is tuple else query))
    db.commit()
    db.close()


def fetch_db(dbname, query_string, return_cursor=False, use_dict_cursor=False, db_conn=None):
    """
    Returns the results of running the `query_string` against the database. If `return_cursor` is `True`
        then the cursor is returned - after execute but before fetching - and any caller who sets
        `return_cursor=True` is expected to later:
        1/ close the cursor

    Optionally, you can ask for a dict cursor which returns a dictionary instead of a tuple so that you can
    specify column names.
    :param query_string:
    :param return_cursor:
    :param use_dict_cursor:
    :param db_conn:
    :return:
    """
    if u"{}" in query_string:
        raise ValueError(u"{{}} found in query string!\n{}".format(query_string))

    try:
        # Sometimes we pass in a db_conn - when the db_conn is passed in, use it, otherwise
        # create a new one
        if not db_conn:
            db_conn = connect(dbname)

        if use_dict_cursor:
            # conn.cursor will return a cursor object, you can use this query to perform queries
            # note that in this example we pass a cursor_factory argument that will
            # dictionary cursor so COLUMNS will be returned as a dictionary so we
            # can access columns by their name instead of index.
            import psycopg2.extras
            cur = db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            cur = db_conn.cursor()
        cur.execute(query_string)
        if return_cursor:
            # Any caller who sets return_cursor=True is expected to later:
            # 1/ close the cursor
            # 2/ call `self._fetch_db_logging(my_debug_index, query_purpose,
            # result_memory_size, start, this_stack_depth)`
            return cur, (debug_idx, query_purpose, None, start, stack_depth)

        res = cur.fetchall()
        cur.close()
        return res
    except Exception, e:
        # Yes - catching everything - but look - re-raising later. This is useful in a multi-processing
        # world where the stacktrace is not otherwise seen.
        traceback.print_exc()
        print u"""The above exception "{}" was raised from the following SQL:\n{}""".format(e.message, query_string)
        raise e


def syntax_checker(queries, dbname):
    if type(queries) is dict:
        queries = queries.values()
    elif type(queries) is not list:
        raise Exception('Bad query collections! Needs to be list or dict!')
    db = connect(dbname)
    for query in queries:
        entries = None
        if type(query) is tuple:
            entries = query[1]
            query = query[0]
        query_replaced = ' '.join(query.lower().replace('\n', '').replace('\t', '').split()).strip().strip(';')
        query_broken = query_replaced.split(' ')
        if query_replaced.startswith('analyze'):
            continue
        if query_replaced.startswith('drop'):
            continue
        if ';' in query_replaced:
            continue
        if query_replaced.startswith('create'):
            if query_broken[1] == 'table' and query_broken[3] == 'as':
                # This is allowed
                pass
            else:
                continue
        try:
            if entries:
                raise Exception('Explain does not work with parameterized queries.')
            else:
                cur = db.cursor()
                cur.execute('EXPLAIN %s' % query)
                cur.close()
        except:
            print '------------------------------------------------'
            print 'This query\'s explain is not working:\n%s' % query
            print '\n------------------------------------------------'
            trace = traceback.format_exc()
            print trace
            print '\n------------------------------------------------'
            print '\n'
            raise Exception('Postgres syntax check failed!')
    db.close()


def purpose_of(sql):
    lines = sql.splitlines()
    if len(lines) == 1:
        p = u"\"{}\"".format(sql.strip())
    else:
        # OK - are there comments here?
        comments = [l for l in lines if l.strip().startswith('--')]
        if comments:
            comments = [l.split('--')[1].strip() for l in comments]
            p = u" ".join(comments)
        else:
            lines = [l.strip() for l in lines]
            p = u"\"{}\"".format(u" ".join(lines))
    return p


def execute_sqls_serially(dbname, sqls, auto_commit=True, spew=False):
    if not sqls:
        return
    if not isinstance(sqls, collections.Iterable):
        raise ValueError(u"`sqls` param must be iterable.")

    db_conn = connect(dbname)
    cur = db_conn.cursor()
    for sql in sqls:
        start = datetime.datetime.utcnow()
        purpose = purpose_of(sql)
        try:
            if spew:
                print u"db will {}".format(purpose)
            cur.execute(sql)
            if auto_commit:
                # This doesn't behave as you'd expect in this scenario :-( db_conn.commit()
                cur.execute("commit;")
        except psycopg2.DatabaseError, e:
            raise type(e), type(e)(e.message.strip() + ' happens for "%s"' % sql), sys.exc_info()[2]

        stop = datetime.datetime.utcnow()
        if spew:
            print u"db spent {} on:  {}".format(str(stop - start).split(".")[0], purpose)


class FuncThread(threading.Thread):
    def __init__(self, target, *args, **kwargs):
        print "FuncThread created: target={} with {} args and {} kwargs".format(str(target), len(args), len(kwargs))
        self._target = target
        self._args = args
        self._kwargs = kwargs
        self.raised_exception = False
        self._result = None
        threading.Thread.__init__(self)

    def run(self):
        # noinspection PyBroadException
        try:
            self._result = self._target(*self._args, **self._kwargs)
        except BaseException:
            self.raised_exception = True
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)

    def get_result(self):
        assert not self.raised_exception
        return self._result


def dataframe_filename(dataframes_dir, name):
    return "{}/{}.json".format(dataframes_dir, name.replace(" ", "_"))


def dataframe_info_filename(dataframes_dir, name):
    return "{}/{}.info".format(dataframes_dir, name.replace(" ", "_"))


# def report_filename(reports_path, name):
#     return "{}/{}.gz64_report".format(reports_path,
#                                       name.lower()
#                                       .replace(" ", "_")
#                                       .replace("-", "_")
#                                       .replace(",", ""))


def save_dataframe(dataframe, dataframes_dir, name, samples=20):

    if not isinstance(dataframe, DataFrameGroupBy):
        with codecs.open(dataframe_info_filename(dataframes_dir, name), mode="wb", encoding="utf8") as f:
            buf = StringIO.StringIO()
            dataframe.info(buf=buf)
            f.write(buf.getvalue())
            buf.close()
            f.write('\n')
            # Warning about the use of .sample: if you request more than there are rows (unlike head and tail) it will
            # throw an exception.
            f.write(dataframe.sample(min(samples, len(dataframe.index))).to_string(justify='left'))
            f.write('\n')

    # Previously this had been pickled - but:
    #
    #     File "/usr/local/lib/python2.7/dist-packages/pandas/core/generic.py", line 1015, in to_pickle
    #       return to_pickle(self, path)
    #     File "/usr/local/lib/python2.7/dist-packages/pandas/io/pickle.py", line 14, in to_pickle
    #       pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
    #   SystemError: error return without exception set
    #
    # This is known to be http://bugs.python.org/issue11872 (which should NOT be "resolved" as it's still a problem.
    #
    # So: the next line cannot be used (safely):
    # dataframe.to_pickle("{}/{}.pickle".format(dataframes_dir, name))
    # pickle.dump(dataframe, open("{}/{}.pickle".format(dataframes_dir, name), "wb"))
    #
    # Alas - the above died - it simply hung and didn't write anything other than the first 486 bytes of the expected
    # 20+ GB.

    # TODO - save pickle when we're reasonably sure it'll save, and adapt this and load to try pickles first, json as a
    # TODO - last resort.
    # There is now the need for a "plan C" which is:
    try:
        dataframe.to_json(dataframe_filename(dataframes_dir, name))
    except Exception, e:
        import logging
        dataframe.to_pickle("{}/{}.emergency-pickle".format(dataframes_dir, name))
        logging.basicConfig(level=logging.DEBUG, filename="{}/{}.emergency-pickle.log".format(dataframes_dir, name))
        logging.error("{}: {}".format(dataframes_dir, name))
        logging.exception(e)
        raise e


def load_dataframe(dataframes_dir, name, ignore_index_range=True, ignore_info=False):
    if 1 < 2:
        raise NotImplementedError()

    dataframe = pd.read_json(dataframe_filename(dataframes_dir, name))
    # Convert Float64Index to Int64Index (which is how I created them in the first place (unless the index is an object)

    # noinspection PyUnresolvedReferences
    if type(dataframe.index) == pd.core.index.Float64Index:
        dataframe = pd.DataFrame(dataframe, index=map(int, dataframe.index.values))

    if not ignore_info:
        buf = StringIO.StringIO()
        dataframe.info(buf=buf)
        read_info = buf.getvalue().split("\n")

        with codecs.open(dataframe_info_filename(dataframes_dir, name), mode="rb", encoding="utf8") as f:
            saved_info = f.readlines()

        # The above `saved_info` is the dataframe info plus a blank line and then the head of the dataframe.
        # To compare the saved with the loaded info, we want to extract the top part:
        saved_df_info = []
        for line in saved_info:
            if line and line.strip():
                saved_df_info.append(line)
            else:
                break

        if ignore_index_range:
            for ix, line in enumerate(saved_df_info):
                if "Index: " in line:
                    saved_df_info[ix] = line.split(", ")[0]
            for ix, line in enumerate(read_info):
                if "Index: " in line:
                    read_info[ix] = line.split(", ")[0]

        expected = "".join(sorted([x.strip() for x in saved_df_info]))
        actual = "".join(sorted([x.strip() for x in read_info]))

        if not expected == actual:
            print dataframes_dir
            print name
            v = u"".join([x.encode(encoding="utf8") for x in saved_df_info])
            print v
            print read_info
            print expected
            print actual
            msg = u"For {}'s {} dataframe, expected:\n" + \
                  "{}\n" + \
                  "--\n" + \
                  "Actual:\n" + \
                  "{}\n" + \
                  "--\ntested ex:\n" + \
                  "{}\n" + \
                  "tested ac:\n" + \
                  "{}\n--".format(
                      unicode(dataframes_dir), unicode(name),
                      u"".join(saved_df_info), unicode(read_info),
                      unicode(expected), unicode(actual))
            raise ValueError(msg)

    return dataframe


def gather_monthly_freqs(monthly_freqs_dir, last=None, num=None):
    if last:
        assert isinstance(last, str) or isinstance(last, unicode)
    if num:
        assert isinstance(num, int)

    fs = {}
    for root, dirs, files in os.walk(monthly_freqs_dir, followlinks=True):
        for f in files:
            if f.endswith(".tsv"):
                rf = os.path.join(root, f)
                if not "/2013/2014-" in rf and not "/2014/2013-" in rf:
                    if last:
                        if f[:len(last)] <= last:
                            fs[rf] = f
                    else:
                        fs[rf] = f
    if not num:
        assert len(fs) > 11, monthly_freqs_dir + " gave us just " + str(len(fs)) + " tsv files."

    if num:
        return [y[0]  # the full path
                for y in sorted(fs.items(),
                                key=lambda x: (x[1], x[0])  # sort by values (filename), then if repeated by keys
                                                            # (the full path, including dir)
                                )[-1 * num:]]
    else:
        return sorted(fs.keys())


def save_gz64_report(reports_path=None, year_ending=None, country_code=None, name=None, desc=None, columns=None,
                     list_of_lists=None):
    assert isinstance(name, (str, unicode))
    assert isinstance(desc, (str, unicode))
    assert isinstance(columns, list)
    assert isinstance(list_of_lists, list)

    data = json.dumps(list_of_lists)
    gz = zlib.compress(data, 9)
    gz_b64 = base64.b64encode(gz)
    filename = "{}.gz64_report".format(name.replace(" ", "_").replace("/", "_").replace(",", "").replace("-", "_"))

    _filename = full_path_and_filename(reports_path, year_ending, country_code, filename)
    with codecs.open(_filename, mode="wb", encoding="utf8") as f:
        for row in [name, desc, pprint.pformat(columns), gz_b64]:
            f.write(row)
            f.write('\n')
        f.write('\n# What follows is a readable copy of the data\n')
        pprint.pprint(list_of_lists, stream=f)
    return _filename


def z64(data, already_dejsoned=False):
    if already_dejsoned:
        return base64.b64encode(zlib.compress(data, 9))
    else:
        return base64.b64encode(zlib.compress(json.dumps(data), 9))



def get_group_stats(group):
    # It's REALLY important that these results are wrapped in round/int/(or float/str though these are not yet
    # catered-for in this code) so that later users don't need to know the name of the series - or worse still that
    # for some - eg mean (which I always round) - doesn't take the series name at all. This might not be the "pythonic"
    # way for treating pandas data, but there you go.
    return {'98%': int(group.quantile(.98)),
            'max': int(group.max()),
            'count': int(group.count()),
            'mean': round(group.mean(), 1),
            'median': int(group.median()),
            'sum': int(group.sum())}


ENGAGEMENT_LEVELS = ["zero calls", "not engaged (0 < c < 10)", "engaged (10 <= c < 100)",
                     "highly engaged (100 <= c < 1000)", "very highly engaged (c >= 1000)"]


def engagement_level(calls):
    """
    CASE
        WHEN calls = 0 THEN '0 - zero calls'
        WHEN calls < 10 THEN '1 - not engaged (0 < c < 10)'
        WHEN calls < 100 THEN '2 - engaged (10 <= c < 100)'
        WHEN calls < 1000 THEN '3 - highly engaged (100 <= c < 1000)'
        ELSE '4 - very highly engaged (c >= 1000)'
    END AS level,
    :return:
    """

    if calls == 0:
        return ENGAGEMENT_LEVELS[0]
    elif calls < 10:
        return ENGAGEMENT_LEVELS[1]
    elif calls < 100:
        return ENGAGEMENT_LEVELS[2]
    elif calls < 1000:
        return ENGAGEMENT_LEVELS[3]
    else:
        return ENGAGEMENT_LEVELS[4]


class ReadWrapper:
    """
    A class which wraps a DataFrame (or any data structure with an itertuples whose lines are a list) into something
    which can be used to populate a database like:

            conn = utils.connect(dbname)
            cur = conn.cursor()
            cur.copy_from(ReadWrapper(<df>), "<table>", null='nan')

    Handy for when you have a worked-up dataframe which you want to put into the database.
    """
    def __init__(self, data, include_index_col=True):
        self.iter = data.itertuples()
        self.include_index_col = include_index_col
        self.has_warned_about_ignoring_size = False

    def readline(self, size=None):
        """
        Reads the next "line" from the iter(ator). The size param is required (it seems to be part of the API for
        such things, but is not used).
        :param size:
        :return:
        """
        if size and not self.has_warned_about_ignoring_size:
            sys.stderr.write("ReadWrapper is ignoring the readline `size` argument value `{}`.\n".format(size))
            sys.stderr.flush()
            self.has_warned_about_ignoring_size = True

        try:
            if self.include_index_col:
                line = self.iter.next()
            else:
                line = self.iter.next()[1:]  # element 0 is the index
            row = '\t'.join(y.encode('utf8') if isinstance(y, unicode) else str(y) for y in line) + '\n'
        except StopIteration:
            return ''
        else:
            return row

    # All "read" calls are to be handled by the readline function
    read = readline


def duration(seconds):
    assert isinstance(seconds, (int, float))
    sec = int(round(seconds))
    if seconds < 100:
        return "{} sec".format(sec)
    if seconds < 60 * 60:
        m, s = divmod(sec, 60)
        return "{} sec ({}m:{:02}s)".format(sec, m, s)
    else:
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return "{} sec ({}h:{:02}m:{:02}s)".format(sec, h, m, s)


def print_full_dataframe(df):
    pd.set_option('display.max_rows', len(df))
    pd.set_option('max_colwidth', 50)
    pd.set_option('display.width', 0)
    print(df)
    pd.reset_option('display.max_rows')
    pd.reset_option('max_colwidth')
    pd.reset_option('display.width')


def copy_to_csvgz(dbname=None, reports_path=None, year_ending=None, country_code=None, sql=None, csv_gz_filename=None):
    """
    Runs the SQL statement `sql` against the database `dbname` and saves the result with filename `csv_gz_filename`.

    It ensures that the filename `csv_gz_filename` ends with ".csv.gz".
    :param dbname:
    :param reports_path:
    :param year_ending:
    :param country_code:
    :param sql:
    :param csv_gz_filename:
    :return: None on success.

    """

    if not csv_gz_filename.endswith(".csv.gz"):
        csv_gz_filename = "{}.csv.gz".format(csv_gz_filename)

    conn = connect(dbname)
    stmt = "COPY ({}) TO STDOUT CSV".format(sql)
    cur = conn.cursor()
    with gzip.open(full_path_and_filename(reports_path,year_ending,country_code, csv_gz_filename),
                   mode="wb", compresslevel=9) as f:
        cur.copy_expert(stmt, f)
    cur.close()
    conn.close()


def full_path_and_filename(reports_path, year_ending, country_code, filename):
    for d in ("{rp}".format(rp=reports_path),
              "{rp}/{ye}".format(rp=reports_path,
                                      ye=year_ending),
              "{rp}/{ye}/{cc}".format(rp=reports_path,
                                      ye=year_ending,
                                      cc="{}/".format(country_code) if country_code else "")):
        if not os.path.exists(d):
            os.makedirs(d)
            assert os.path.exists(d)
            assert os.access(d, os.W_OK)

    if filename:
        return "{rp}/{ye}/{cc}{fn}".format(rp=reports_path,
                                           ye=year_ending,
                                           cc="{}/".format(country_code) if country_code else "",
                                           fn=filename)
    else:
        return "{rp}/{ye}/{cc}".format(rp=reports_path,
                                       ye=year_ending,
                                       cc="{}/".format(country_code) if country_code else "")


if __name__ == '__main__':
    print duration(7258.61051798)
    print 'Nothing to do.'


