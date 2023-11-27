Installation
============

IPS is available on PyPI and can be installed using

..  code-block:: bash

    pip install ipsuite


Note that a default installation does not come with any of the models.
These are available as extras and can be installed by specifying the models you would like to install.
For example, the following code snippets install the GAP model (via `quippy <https://libatoms.github.io/QUIP/#>`_ ) and all available models respectively

..  code-block:: bash

    pip install ipsuite[gap]

..  code-block:: bash

    pip install ipsuite[all]



Developer Installation
----------------------

IPS is devloped using `Poetry <https://python-poetry.org/>`_.
To install a developer verions of IPS, clone the repository and install it with poetry:

..  code-block:: bash

    git clone https://github.com/zincware/IPSuite.git
    cd IPSuite
    poetry install
