import codecs
from bs4 import BeautifulSoup
import datetime
from eutils import connect


def load_device(root, dbname, xml_file):
    """

    :param root:
    :param dbname:
    :param xml_file:
    :return:
    """
    bgdata = root.find("bgdata")
    assert bgdata
    attrss = []
    for bg in bgdata.find_all("bg"):
        # <BG Val="13.8" Dt="2015-09-26" Tm="21:35" Flg="M1" Carb="25" D="1"/>
        attrs = bg.attrs

        if attrs["val"] == "---":
            attrs["val"] = None
        elif attrs["val"] == "HI":
            attrs["val"] = 30.0
        else:
            attrs["val"] = float(attrs["val"])
        if attrs["val"] and attrs["val"] > 30.0:
            attrs["val"] = 30.0

        attrs["_source_file"] = xml_file

        if not "carb" in attrs:
            attrs["carb"] = None
        else:
            attrs["carb"] = int(attrs["carb"])

        for a in ("flg", "evt", "ins1"):
            if not a in attrs:
                attrs[a] = None

        attrss.append(attrs)

    sql_tmpl = u"""
        INSERT INTO load_smartpix_device (
            _source_file,
            val,
            dt,
            tm,
            flg,
            evt,
            ins1,
            carb,
            d
        ) VALUES (
            %(_source_file)s,
            %(val)s,
            %(dt)s,
            %(tm)s,
            %(flg)s,
            %(evt)s,
            %(ins1)s,
            %(carb)s,
            %(d)s
        )
    """
    with connect(dbname) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(sql_tmpl, attrss)
        conn.commit()


def load_ip(root, dbname, xml_file):
    """

    :param root:
    :param dbname:
    :param xml_file:
    :return:
    """
    ipdata = root.find("ipdata")
    assert ipdata

    basals = []
    for basal in ipdata.find_all("basal"):
        attrs = basal.attrs
        attrs["_source_file"] = xml_file
        attrs["cbrf"] = float(attrs["cbrf"])
        for a in ("profile", "tbrdec", "tbrinc", "cmd", "remark"):
            if not a in attrs:
                attrs[a] = None
        basals.append(attrs)

    sql_tmpl = u"""
        INSERT INTO load_smartpix_basal (
            _source_file,
            dt,
            tm,
            cbrf,
            tbrdec,
            tbrinc,
            profile,
            cmd,
            remark
        ) VALUES (
            %(_source_file)s,
            %(dt)s,
            %(tm)s,
            %(cbrf)s,
            %(tbrdec)s,
            %(tbrinc)s,
            %(profile)s,
            %(cmd)s,
            %(remark)s
        )
    """
    with connect(dbname) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(sql_tmpl, basals)
        conn.commit()

    boluses = []
    for bolus in ipdata.find_all("bolus"):
        attrs = bolus.attrs
        attrs["_source_file"] = xml_file
        attrs["amount"] = float(attrs["amount"])
        for a in ("type", "cmd", "remark"):
            if not a in attrs:
                attrs[a] = None

        boluses.append(attrs)

    sql_tmpl = u"""
        INSERT INTO load_smartpix_bolus (
            _source_file,
            dt,
            tm,
            amount,
            "type",
            cmd,
            remark
        ) VALUES (
            %(_source_file)s,
            %(dt)s,
            %(tm)s,
            %(amount)s,
            %(type)s,
            %(cmd)s,
            %(remark)s
        )
    """
    with connect(dbname) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(sql_tmpl, boluses)
        conn.commit()

    events = []
    for event in ipdata.find_all("event"):
        attrs = event.attrs
        attrs["_source_file"] = xml_file
        for a in ("shortinfo", "description"):
            if not a in attrs:
                attrs[a] = None

        events.append(attrs)

    sql_tmpl = u"""
        INSERT INTO load_smartpix_event (
            _source_file,
            dt,
            tm,
            shortinfo,
            description
        ) VALUES (
            %(_source_file)s,
            %(dt)s,
            %(tm)s,
            %(shortinfo)s,
            %(description)s
        )
    """
    with connect(dbname) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(sql_tmpl, events)
        conn.commit()

def load(xml_file, dbname=None):
    """

    :param xml_file:
    :param dbname:
    :return:
    """
    with codecs.open(xml_file, encoding="ISO-8859-1") as fff:
        soup = BeautifulSoup(fff, "lxml")
        root = soup.find("import")
        if not root:
            # Some non-SmartPix file (eg .DS_Store)
            return

        # Is this the meter (device) or the pump (ip)?
        device = root.find("device")
        ip = root.find("ip")

        if device:
            assert not ip
        elif ip:
            assert not device
        assert device or ip

        print "        - parsing {}".format(xml_file)
        if device:
            load_device(root, dbname, xml_file)
        else:
            load_ip(root, dbname, xml_file)
