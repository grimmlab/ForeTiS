HowTo: Run ForeTiS using Docker
======================================
We assume that you succesfully did all steps described in :ref:`Docker workflow`: to setup ForeTiS using Docker.

Workflow
"""""""""""
You are at the **root directory within your Docker container**, i.e. after step 5 of the setup at :ref:`Docker workflow`:.

If you closed the Docker container you created at the end of the installation, just use ``docker start -i CONTAINERNAME``
to start it in interactive mode again. If you did not create a container yet, go back to step 5 of the setup.

1. Navigate to the directory where the ForeTiS repository is placed within your container

    .. code-block::

        cd /REPO_DIRECTORY/IN/CONTAINER/ForeTiS

2. Run ForeTiS (as module). By default, ForeTiS starts the optimization procedure for 10 trials with XGBoost and a 5-fold nested cross-validation using the data we provide in ``tutorials/tutorial_data``.

    .. code-block::

        python3 -m ForeTiS.run --save_dir SAVE_DIRECTORY

    That's it! Very easy! You can now find the results in the save directory you specified.

3. To get an overview of the different options you can set for running ForeTiS, just do:

    .. code-block::

        python3 -m ForeTiS.run --help


Feel free to test ForeTiS, e.g. with other prediction models.
If you want to start using your own data, please carefully read our :ref:`Data Guide`: to ensure that your data fulfills all requirements.