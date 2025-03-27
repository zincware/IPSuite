Geometry Relaxation
===================

In this example we will setup a simulation box and relax the structure using an :term:`MLIP`.

.. tip::

    Setup a new project in an empty directory by calling

    .. code::

        git init
        dvc init
        touch main.py


Add the following to your main.py file:

.. code::

    import ipsuite as ips

    project = ips.Project()
    # create the main project to handle the workflow

    with project:
    # add new nodes to the project within the project context manager
        ips.Smiles2Conformers(smiles="O", numConfs=1) # add our first node

    project.build()

To build the project, run the file using :code:`python main.py`.

.. note::
    
    This will construct the workflow, but not execute.

To execute the workflow now constructed, run :code:`dvc repro`.


.. code::

    import ipsuite as ips

    project = ips.Project()
    # create the main project to handle the workflow

    with project:
    # add new nodes to the project within the project context manager
        water = ips.Smiles2Conformers(smiles="O", numConfs=1) # add our first node
        ammonia = ips.Smiles2Conformers(smiles="N", numConfs=1) # add another node
        box = ips.MultiPackmol(
            data=[water.frames, ammonia.frames],
            count=[10, 3],
            density=997,
            n_configurations=3
        )

    project.build()