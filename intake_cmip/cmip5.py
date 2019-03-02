#!/usr/bin/env python
import os
from collections import OrderedDict

import intake_xarray.base
import numpy as np
import pandas as pd
import xarray as xr

from ._version import get_versions
from .config import glade_cmip5_db

__version__ = get_versions()["version"]
del get_versions


class CMIP5DataSource(intake_xarray.base.DataSourceMixin):
    """ Read CMIP5 data sets into xarray datasets

    """

    container = "xarray"
    version = __version__
    partition_access = True
    name = "cmip5"

    def __init__(self, database, metadata=None):

        """

        Parameters
        ----------

        database : string or file handle
             File path or object for cmip5 database. For users with access to
             NCAR's glade file system, this argument can be set to 'glade'.
        """

        # store important kwargs
        self.database = self._read_database(database)
        self.urlpath = ""
        self.query = {}
        self.query_results = None
        self._ds = None
        super(CMIP5DataSource, self).__init__(metadata=metadata)

    def _read_database(self, database):
        if database == "glade":
            database = glade_cmip5_db
        if os.path.exists(database):
            return pd.read_csv(database)
        else:
            raise FileNotFoundError(f"{database}")

    def _open_dataset(self):

        ens_filepaths = self._get_ens_filepaths()

        ds_list = [xr.open_mfdataset(paths) for paths in ens_filepaths.values()]
        ens_list = list(ens_filepaths.keys())
        self._ds = xr.concat(ds_list, dim="ensemble")
        self._ds["ensemble"] = ens_list

    def to_xarray(self, dask=True):
        """Return dataset as an xarray instance"""
        if dask:
            return self.to_dask()
        return self.read()

    def search(
        self,
        model=None,
        experiment=None,
        frequency=None,
        realm=None,
        ensemble=None,
        varname=None,
    ):

        """
        Parameters
        -----------

        model : str
              identifies the model used (e.g. HADCM3, HADCM3-233).
        experiment : str
             identifies either the experiment or both the experiment family and a specific type
             within that experiment family.
        frequency : str
            indicates the interval between individual time-samples in the atomic dataset.
            For CMIP5, the following are the only options:

            - yr
            - mon
            - day
            - 6hr
            - 3hr
            - subhr
            - monClim
            - fx

        realm : str
             indicates which high level modeling component is of particular relevance for
             the dataset. For CMIP5, permitted values are:

             - atmos
             - ocean
             - land
             - landIce
             - seaIce
             - aerosol
             - atmosChem
            - ocnBgchem
        ensemble : str
            (r<N>i<M>p<L>): This triad of integers (N, M, L), formatted as (e.g., “r3i1p21”)
            distinguishes among closely related simulations by a single model.
            All three are required even if only a single simulation is performed.
        varname : str, optional
             Variable name according to CMIP Data Reference Syntax (DRS)

        """

        self.query = {
            "model": model,
            "experiment": experiment,
            "frequency": frequency,
            "realm": realm,
            "ensemble": ensemble,
            "varname": varname,
        }
        database = self.database
        condition = np.ones(len(database), dtype=bool)

        for key, val in self.query.items():
            if val is not None:

                condition = condition & (database[key] == val)

        self.query_results = database.loc[condition]
        return self

    def results(self):
        return self.query_results

    def _get_ens_filepaths(self):
        if self.query_results.empty:
            raise ValueError(
                f"No dataset found for:\n \
                                  \tmodel = {self.query['model']}\n \
                                  \texperiment = {self.query['experiment']} \n \
                                  \tfrequency = {self.query['frequency']} \n \
                                  \trealm = {self.query['realm']} \n \
                                  \tensemble = {self.query['ensemble']} \n \
                                  \tvarname = {self.query['varname']}"
            )

        models = self.query_results.ensemble.nunique() > 1
        experiments = self.query_results.experiment.nunique() > 1
        frequencies = self.query_results.frequency.nunique() > 1

        if models or experiments or frequencies:

            raise ValueError(
                f"Invalid results for search query = {self.query}.\n\
                              Please specify unique model, experiment, and frequency to use"
            )

        # Check that the same varname is not in multiple realms
        realm_list = self.query_results.realm.unique()
        if len(realm_list) != 1:
            raise ValueError(
                f"{self.query['varname']} found in multiple realms:\
                  \t{self.query['realm_list']}. Please specify the realm to use"
            )

        ds_dict = OrderedDict()
        for ens in self.query_results["ensemble"].unique():
            ens_match = self.query_results["ensemble"] == ens
            paths = self.query_results.loc[ens_match]["file_fullpath"].tolist()
            ds_dict[ens] = paths

        return ds_dict
