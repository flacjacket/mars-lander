# How to run this notebook

This notebook details the theory and setup of the neural network that has been
implemented to determine the landing calculations.  Here, I layout the basic
things you need to get this notebook up and running.

I note that I have coded this with Python version compatibility in mind, but I
run on Python 3.4, I cannot guarantee that it will run correctly on previous
Python versions.

# Installing libraries

In addition to the standard scientific Python stack (NumPy, SciPy, matplotlib),
you will need to install [pylearn2](https://github.com/lisa-lab/pylearn2).  The
hard part of this is installing [Theano](https://github.com/Theano/Theano), a
library that allows for very efficient evaluation of array expressions,
including C code generation and transparent GPU operation. To install Theano
will require installing header files and compiling C extensions, check the
[docs](http://deeplearning.net/software/theano/install.html) for all of the
requirements. Once everything is in place, you should be able to `pip install
Theano`. This is fairly straightforward with Linux, for Windows, I would
recommend installing the prerequisites through Anaconda and just pip
installing.

Next, you can install pylearn2.  This library does not make releases or push
anything up to PyPI, so you will have to install from the latest git source. In
order to do that, with a working git install, you will need to pull their
[Github repo](https://github.com/lisa-lab/pylearn2) and install from there. To
do this:
```bash
git clone git://github.com/lisa-lab/pylearn2.git
python setup.py install
```
More information is detailed in the
[docs](http://deeplearning.net/software/pylearn2/#download-and-installation).

# Provided files

In this git repository, I have provided a host of Python scripts and utilities
that I used when I was developing this algorithm. Much of the scripting has
been reproduced and cleaned up in the notebook, but some of the utilities, in
particular for reading and writing the appropriate file formats, are imported
in this notebook. If you have done a git clone of this repository, these files
will be in the correct location to be found when running the notebook.

In addition, there are a couple things that I use in this notebook for graphics
and data presentation. One of these, `tikzmagick.py` is an IPython magic that
allows tikz vector graphics to be rendered in the notebook. I provide the
magic, and if you just open the notebook, all of the tikz cells have been
rendered (you may have to "Trust" the notebook to see the pre-rendered
versions); however, if you wish to run these cells, you will also need to have
a working latex compiler installed and in your path. The tikz magic is BSD
licensed, and available [here](http://sourceforge.net/projects/pgf/).

I have also included a custom set of color maps to use to show the height data.
I will stress here: **DO NOT USE JET**. If you want to see why, you can check
out the notebook, as I highlight some of the terrible things jet can do to your
data.  The included color maps, viridis, magma, plasma, and inferno (and their
reversed counterparts) are all provided.  All of these will be available in the
next major release of matplotlib, 2.0, and viridis will replace jet as the
default.
