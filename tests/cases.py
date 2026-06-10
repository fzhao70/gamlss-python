"""Model-fit case registry mirroring r-scripts/gen_reference.R exactly."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import gamlss as gl

HERE = os.path.dirname(__file__)
REFDATA = os.path.join(HERE, "reference", "data")


def load_case_data(name):
    """Load a dataset by name: bundled gamlss.data or simulated CSV."""
    path = os.path.join(REFDATA, f"{name}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return gl.load_data(name)


def weights_int():
    return pd.read_csv(os.path.join(REFDATA, "weights_int.csv"))["w"].to_numpy()


# name -> spec; mirrors gen_reference.R
CASES = {
    "no_abdom": dict(formula="y ~ x", family="NO", data="abdom"),
    "no_abdom_sx": dict(formula="y ~ x", sigma_formula="~x", family="NO",
                        data="abdom"),
    "no_abdom_poly": dict(formula="y ~ poly(x, 3)", sigma_formula="~x",
                          family="NO", data="abdom"),
    "ga_abdom": dict(formula="y ~ x", sigma_formula="~x", family="GA",
                     data="abdom"),
    "logno_abdom": dict(formula="y ~ x", family="LOGNO", data="abdom"),
    "ig_abdom": dict(formula="y ~ x", family="IG", data="abdom"),
    "gu_abdom": dict(formula="y ~ x", family="GU", data="abdom"),
    "rg_abdom": dict(formula="y ~ x", family="RG", data="abdom"),
    "lo_abdom": dict(formula="y ~ x", family="LO", data="abdom"),
    "wei_abdom": dict(formula="y ~ x", family="WEI", data="abdom"),
    "wei3_abdom": dict(formula="y ~ x", family="WEI3", data="abdom"),
    "tf_abdom": dict(formula="y ~ x", sigma_formula="~x", family="TF",
                     data="abdom"),
    "pe_abdom": dict(formula="y ~ x", family="PE", data="abdom", n_cyc=200),
    "bccg_abdom": dict(formula="y ~ x", sigma_formula="~x", family="BCCG",
                       data="abdom", n_cyc=200),
    "bct_abdom": dict(formula="y ~ x", sigma_formula="~x", family="BCT",
                      data="abdom", n_cyc=200),
    "bcpe_abdom": dict(formula="y ~ x", sigma_formula="~x", family="BCPE",
                       data="abdom", n_cyc=200),
    "bcto_abdom": dict(formula="y ~ x", sigma_formula="~x", family="BCTo",
                       data="abdom", n_cyc=200),
    "no_abdom_cg": dict(formula="y ~ x", sigma_formula="~x", family="NO",
                        data="abdom", method=lambda: gl.CG()),
    "bct_abdom_mixed": dict(formula="y ~ x", sigma_formula="~x",
                            family="BCT", data="abdom", n_cyc=200,
                            method=lambda: gl.mixed(2, 50)),
    "no_abdom_w": dict(formula="y ~ x", family="NO", data="abdom",
                       weights="int"),
    "ga_usair": dict(formula="y ~ x1 + x2 + x3 + x4 + x5 + x6", family="GA",
                     data="usair"),
    "ga_rent": dict(formula="R ~ Fl + A + C(H) + C(loc)",
                    sigma_formula="~Fl", family="GA", data="rent"),
    "po_aids": dict(formula="y ~ x + C(qrt)", family="PO", data="aids"),
    "nbi_aids": dict(formula="y ~ x + C(qrt)", family="NBI", data="aids"),
    "po_fabric": dict(formula="y ~ x", family="PO", data="fabric"),
    "nbi_sim": dict(formula="y ~ x1 + x2", sigma_formula="~x1", family="NBI",
                    data="sim_nbi"),
    "nbii_sim": dict(formula="y ~ x1", family="NBII", data="sim_nbii"),
    "geom_sim": dict(formula="y ~ x1", family="GEOM", data="sim_geom"),
    "pig_sim": dict(formula="y ~ x1", family="PIG", data="sim_pig"),
    "zip_sim": dict(formula="y ~ x1", family="ZIP", data="sim_zip"),
    "zip2_sim": dict(formula="y ~ x1", family="ZIP2", data="sim_zip2"),
    "zinbi_sim": dict(formula="y ~ x1", family="ZINBI", data="sim_zinbi",
                      n_cyc=200),
    "zanbi_sim": dict(formula="y ~ x1", family="ZANBI", data="sim_zanbi",
                      n_cyc=200),
    "bi_sim": dict(formula="cbind(y, fail) ~ x1", family="BI",
                   data="sim_bi"),
    "bb_sim": dict(formula="cbind(y, fail) ~ x1", family="BB",
                   data="sim_bb"),
    "zabi_sim": dict(formula="cbind(y, fail) ~ x1", family="ZABI",
                     data="sim_zabi", n_cyc=200),
    "zibi_sim": dict(formula="cbind(y, fail) ~ x1", family="ZIBI",
                     data="sim_zibi", n_cyc=200),
    "be_sim": dict(formula="y ~ x1", family="BE", data="sim_be"),
    "beo_sim": dict(formula="y ~ x1", family="BEo", data="sim_beo"),
    "jsu_sim": dict(formula="y ~ x", family="JSU", data="sim_jsu",
                    n_cyc=200),
    "gg_sim": dict(formula="y ~ x1", family="GG", data="sim_gg", n_cyc=200),
    "shasho_sim": dict(formula="y ~ 1", family="SHASHo", data="sim_shasho",
                       n_cyc=200),
    "exp_rent": dict(formula="R ~ Fl + A", family="EXP", data="rent"),
}


def fit_case(name):
    """Fit the Python model for a registry case; returns the result."""
    spec = CASES[name]
    data = load_case_data(spec["data"])
    fam = getattr(gl.dist, spec["family"])()
    kwargs = dict(trace=False)
    if "n_cyc" in spec:
        kwargs["n_cyc"] = spec["n_cyc"]
    if spec.get("method"):
        kwargs["method"] = spec["method"]()
    w = None
    if spec.get("weights") == "int":
        w = weights_int()
    m = gl.gamlss(
        spec["formula"],
        sigma_formula=spec.get("sigma_formula", "~1"),
        nu_formula=spec.get("nu_formula", "~1"),
        tau_formula=spec.get("tau_formula", "~1"),
        family=fam,
        data=data,
        weights=w,
        **kwargs,
    )
    return m, data
