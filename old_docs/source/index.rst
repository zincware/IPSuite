.. IPSuite documentation master file, created by
   sphinx-quickstart on Mon May 22 19:58:40 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

IPS - The Inter Atomic Potential Suite
======================================

The Inter Atomic Potential Suite (IPS) allows for the fast construction and extension of atomistic machine learning workflows.
It is based on `ZnTrack <https://github.com/zincware/ZnTrack/>`_ and thus allows for fully version controlled and reproducible workflows.
Within IPS, we provide numerous Nodes for creating atomistic data, training models, analyzing predictions and performing molecular dynamics.



Example
=======

Routine workflows are easy to set up and can be easily extended for more complex tasks.
Training any of the interfaced models only requires a few lines of code:

.. code-block:: python

   import ipsuite as ips

   with ips.Project() as project:
      data = ips.AddData(file="dataset.extxyz")
      random_selection = ips.configuration_selection.RandomSelection(
          data=data, n_configurations=100
      )

      model = ips.models.GAP(data=random_selection)



.. toctree::
   :hidden:

   _get_started/index
   _examples/index
   _nodes/index

