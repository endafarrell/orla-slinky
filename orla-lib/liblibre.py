import pandas as pd
from eutils import connect

Libre_RecordType_0 = "CGM"
Libre_RecordType_1 = "Scan"
Libre_RecordType_2 = None
Libre_RecordType_3 = None
Libre_RecordType_4 = "Insulin"
Libre_RecordType_5 = "Carbs"
Libre_RecordType_6 = "Time update"


def load(txt_file, dbname=None):
    df = pd.read_csv(txt_file, sep="\t", skiprows=2)

    # Names with open and close brackets, "(" and ")", break the use of executemany. It's easiest to rename them
    # (and this is done in the schema step too).
    columns = dict(zip(df.columns, df.columns))
    for c in columns:
        columns[c] = c.replace("(", "[").replace(")", "]")
    df.rename(index=str, columns=columns, inplace=True)

    # In the DB, I'd like to have empty fields be null rather than "NaN", so:
    df = df.where((pd.notnull(df)), None)

    # We'd like a list of dicts for insertion in bulk to the database
    as_dicts = df.to_dict(orient="record")

    # Upgrade with info about where this came from
    for d in as_dicts:
        d["_source_file"] = txt_file

    with connect(dbname) as conn:
        with conn.cursor() as cursor:
            assert 'Historic Glucose [mmol/L]' in as_dicts[0], as_dicts[0].keys()
            # There are a lot of columns ...
            sql_tmpl = u"""
        INSERT INTO load_libre (
            "_source_file",
            "ID",
            "Time",
            "Record Type",
            "Historic Glucose [mmol/L]",
            "Scan Glucose [mmol/L]",
            "Non-numeric Rapid-Acting Insulin",
            "Rapid-Acting Insulin [units]",
            "Non-numeric Food",
            "Carbohydrates [grams]",
            "Non-numeric Long-Acting Insulin",
            "Long-Acting Insulin [units]",
            "Notes",
            "Strip Glucose [mmol/L]",
            "Ketone [mmol/L]",
            "Meal Insulin [units]",
            "Correction Insulin [units]",
            "User Change Insulin [units]",
            "Previous Time",
            "Updated Time"
        ) VALUES (
            %(_source_file)s,
            %(ID)s,
            %(Time)s,
            %(Record Type)s,
            %(Historic Glucose [mmol/L])s,
            %(Scan Glucose [mmol/L])s,
            %(Non-numeric Rapid-Acting Insulin)s,
            %(Rapid-Acting Insulin [units])s,
            %(Non-numeric Food)s,
            %(Carbohydrates [grams])s,
            %(Non-numeric Long-Acting Insulin)s,
            %(Long-Acting Insulin [units])s,
            %(Notes)s,
            %(Strip Glucose [mmol/L])s,
            %(Ketone [mmol/L])s,
            %(Meal Insulin [units])s,
            %(Correction Insulin [units])s,
            %(User Change Insulin [units])s,
            %(Previous Time)s,
            %(Updated Time)s
        ); """

            cursor.executemany(sql_tmpl, as_dicts)
        conn.commit()