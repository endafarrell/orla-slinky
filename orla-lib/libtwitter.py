import pandas as pd
import codecs
import json
from eutils import connect


def load(jsons_file, dbname):

    with codecs.open(jsons_file, encoding="utf8") as f:
        df = pd.DataFrame(json.loads(line) for line in f)
    if not len(df.index):
        return
    df["_source_file"] = jsons_file

    # We'd like a list of dicts for insertion in bulk to the database
    as_dicts = df.to_dict(orient="record")

    # There are a lot of columns ...
    sql_tmpl = u"""
        INSERT INTO load_twitter (
            "_source_file",
            "id",
            "created_at",
            "recipient_id",
            "recipient_screen_name",
            "sender_id",
            "sender_screen_name",
            "text"
        ) VALUES (
            %(_source_file)s,
            %(id)s,
            %(created_at)s,
            %(recipient_id)s,
            %(recipient_screen_name)s,
            %(sender_id)s,
            %(sender_screen_name)s,
            %(text)s
        ); """

    with connect(dbname) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(sql_tmpl, as_dicts)
        conn.commit()