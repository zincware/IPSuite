.. _quickstart:

Quickstart
==========

.. image:: https://raw.githubusercontent.com/zincware/IPSuite/main/misc/IPS_logo.png
    :width: 800
    :alt: IPS Logo

What is IPS?
------------

IPS (or Interatomic Potentials Suite) is a tool for working with Machine Learned Interatomic Potentials 
and construcing computational workflows which are executed at a later time.
Using `DVC <https://dvc.org/>`_ as its backbone, IPS allows efficent and transparent data versioning of complex datasets.

Installation
------------

Download `the latest IPS relase <https://github.com/zincware/IPSuite>`_  from GitHub or install the IPS package using pip

.. code-block:: bash

    pip install ipsuite

create a folder for your project and initialize git as well as DVC to start working with IPS.

.. code-block:: bash
    
    mkdir project
    cd project
    git init
    dvc init

In a file :code:`main.py` try importing IPS to check if everything is working correctly

.. code-block:: python

    import ipsuite as ips


Getting started
---------------
A simple example




The IPS Project
---------------

.. code-block:: python

    with ips.Project() as project
        ...


Creating Experiments
--------------------




Custom Nodes
------------






