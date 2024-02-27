.. _getting_started:

Getting started
===============

Installation
------------

The IPS package can be downloaded via the package-management system pip. 
We recomend the usage of an python environment with the python version 3.10. Other versions might not currently work.

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

Anther way to install IPSuite is to download the `latest version from GitHub <https://github.com/zincware/IPSuite>`_ 
and using `Poetry <https://python-poetry.org/>`_ to install the python dependencies.

.. code-block:: bash

    git clone https://github.com/zincware/IPSuite.git
    cd IPSuite
    poetry install .

The you can create another folder for your own project.

.. code-block:: bash
    
    mkdir project
    cd project
    git init
    dvc init

To get to know the IPS procedures we will do an example problem.

The First Project
=================

Initial Data Generation
-----------------------
This project will show the basic creation of an IPS Project using an additional piece of software
called `Packmol <https://m3g.github.io/packmol/>`_.

.. code-block:: python

    import ipsuite as ips

    with ips.Project() as project:
        mol = ips.configuration_generation.SmilesToAtoms(smiles="O")
        packmol = ips.configuration_generation.Packmol(data=[mol.atoms], count=[10], density=876)

    project.build()





Creating Experiments
--------------------