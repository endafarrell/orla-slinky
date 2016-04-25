import pandas as pd
import codecs
import json
from eutils import connect


def load(jsons_file, dbname):

    with codecs.open(jsons_file, encoding="utf8") as f:
        df = pd.DataFrame(json.loads(line) for line in f if not line.startswith("//"))
    if not len(df.index):
        return
    df["_source_file"] = jsons_file

    # We'd like a list of dicts for insertion in bulk to the database
    as_dicts = df.to_dict(orient="record")

    # There are a lot of columns ...
    sql_tmpl = u"""
        INSERT INTO load_runkeeper (
            _source_file,
            uri,
            created_at,
            duration,
            "type",
            total_distance
        ) VALUES (
            %(_source_file)s,
            %(uri)s,
            %(start_time)s,
            %(duration)s,
            %(type)s,
            %(total_distance)s
        ); """

    with connect(dbname) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(sql_tmpl, as_dicts)
        conn.commit()