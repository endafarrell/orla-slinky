use the anaconda distribution
-----------------------------
I've given up on pip and co: instead I use conda

    chmod +x Miniconda-latest-Linux-x86_64.sh
    ./Miniconda-latest-Linux-x86_64.sh
    conda env create --file orla.yml
    source activate orla

run a complete build thus:
--------------------------
Usually, it's best to rerun everything:

    ./slinky --spew --verbose  --set config_twitter=~/twitter.auth.yaml drop load

You'll notice that the twitter config - which has my app's credentials - are not part of this code and is loaded from an
external file. Right now it's necessary if you use Twitter DMs to add notes to the data. If you're rapidly prototyping
and/or testing, you might want to avoid Twitter's rate limiting by not sourcing new DMs:

    ./slinky --spew --verbose  --set config_twitter=~/twitter.auth.yaml --set skip_source_twitter=True drop transform


