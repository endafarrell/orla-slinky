#!/usr/bin/env python
from flask import Flask, abort, request
from uuid import uuid4
import requests
import requests.auth
import urllib

# PERSONAL TO THE ORLA APP:
# https://runkeeper.com/partner/applications/keysAndURLs?clientID=68a5efbf62ae41e9bc3b93676ad30189
CLIENT_ID = "68a5efbf62ae41e9bc3b93676ad30189"
CLIENT_SECRET = "ef1d6633e97b4e7880e712ba97ec55b5"
LOCAL_PORT = 65010
LOCAL_CALLBACK = "/rk_callback"
REDIRECT_URI = "http://localhost:{}{}".format(LOCAL_PORT, LOCAL_CALLBACK)


# RUNKEEPER SETTINGS
AUTHORIZATION_URL = "https://runkeeper.com/apps/authorize"
ACCESS_TOKEN_URL = "https://runkeeper.com/apps/token"
DE_AUTHORISATION_URL = "https://runkeeper.com/apps/de-authorize"
CONTENT_TYPE_PROFILE = "application/vnd.com.runkeeper.Profile+json"
CONTENT_TYPE_USER = "application/vnd.com.runkeeper.User+json"
SERVICE_ROOT = "https://api.runkeeper.com"

def user_agent():
    return "oauth2-orla"

def base_headers():
    return {"User-Agent": user_agent()}


app = Flask(__name__)
@app.route('/')
def homepage():
    text = '<a href="%s">Authenticate with Runkeeper</a>'
    return text % make_authorization_url()


def make_authorization_url():
    # Generate a random string for the state parameter
    # Save it for use later to prevent xsrf attacks
    state = str(uuid4())
    save_created_state(state)
    params = {"client_id": CLIENT_ID,
              "response_type": "code",
              "state": state,
              "redirect_uri": REDIRECT_URI,
              "duration": "temporary",
              "scope": "identity"}
    url = "{}?{}".format(AUTHORIZATION_URL, urllib.urlencode(params))
    return url


# Left as an exercise to the reader.
# You may want to store valid states in a database or memcache.
def save_created_state(state):
    pass
def is_valid_state(state):
    return True

@app.route(LOCAL_CALLBACK)
def runkeeper_callback():
    error = request.args.get('error', '')
    if error:
        return "Error: " + error
    state = request.args.get('state', '')
    if not is_valid_state(state):
        # Uh-oh, this request wasn't started by us!
        abort(403)
    code = request.args.get('code')
    print "runkeeper_callback called, got code `{}`".format(code)
    access_token = get_token(code)
    print "runkeeper_callback now seems to have access token `{}`".format(access_token)

    # Note: In most cases, you'll want to store the access token, in, say,
    # a session for use in other parts of your web app.
    #return "Your Runkeeper username is: %s" % get_username(access_token)
    return "Your Runkeeper user ID is {}".format(get_userid(access_token))


def get_token(code):
    post_data = {"grant_type": "authorization_code",
                 "code": code,
                 "redirect_uri": REDIRECT_URI,
                 "client_id": CLIENT_ID,
                 "client_secret": CLIENT_SECRET}
    headers = base_headers()
    response = requests.post(ACCESS_TOKEN_URL,
                             headers=headers,
                             data=post_data)
    import pprint
    pprint.pprint(response)
    token_json = response.json()
    return token_json["access_token"]


def get_username(access_token):
    headers = base_headers()
    headers.update({"Accept": CONTENT_TYPE_PROFILE})
    response = requests.get("{}/profile?access_token={}".format(SERVICE_ROOT, access_token), headers=headers)
    content = response.content
    print "get_username: response.status_code = {}".format(response.status_code)
    print "get_username: response.content = {}".format(content)
    assert response.ok, content
    me_json = response.json()
    try:
        return me_json['name']
    except KeyError, e:
        import pprint
        assert False, pprint.pformat(me_json)

def get_userid(access_token):
    headers = base_headers()
    headers.update({"Accept": CONTENT_TYPE_USER})
    response = requests.get("{}/user?access_token={}".format(SERVICE_ROOT, access_token), headers=headers)
    content = response.content
    print "get_userid: response.status_code = {}".format(response.status_code)
    print "get_userid: response.content = {}".format(content)
    assert response.ok, content
    me_json = response.json()
    try:
        return me_json['userID']
    except KeyError, e:
        import pprint
        assert False, pprint.pformat(me_json)

if __name__ == '__main__':
    get_username("9f5d9328071843eea4ef20add0f229cc")
    app.run(debug=True, port=LOCAL_PORT)