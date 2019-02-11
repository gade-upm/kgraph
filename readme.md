# Configure enviroment 

**Install Anaconda**

https://docs.anaconda.com/anaconda/install/

**Python 3.7**
* conda create -n kgraph --yes python=3.7
* source activate kgraph
* conda install --yes numpy
* conda install -c conda-forge --yes randomcolor
* pip install dengraph
* conda install -c conda-forge -c pkgw-forge -c ostrokach-forge --yes graph-tool
* sudo gdk-pixbuf-query-loaders --update-cache

**Python 2.7**
* conda create -n kgraph --yes python=2.7.9
* source activate kgraph
* conda install --yes numpy=1.10.4
* conda install -c conda-forge --yes randomcolor
* pip install dengraph
* conda install -c conda-forge -c pkgw-forge -c ostrokach-forge --yes graph-tool
* sudo gdk-pixbuf-query-loaders --update-cache

# Check enviroment

* python manage.py -a kores -i data/netscience.graphml -k 5 -comp
* python manage.py -a dengraph -eps 1 -mu 1 -i data/netscience.graphml -k 5 -comp
* python manage.py -h

# Control enviroment

**Activate**
* source activate kgraph

**Deactivate** 
* source deactivate kgraph


**Remove env**
* source deactivate kgraph
* conda remove -n kgraph --all
* conda env remove -n kgraph