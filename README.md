# Comparison of DBMs, VAEs, GANs and multivariate imputation as generative models for genetic variant data

This repository contains an experiment that compares deep Boltzmann machines (DBMs), variational autoencoders (VAEs), generative adversarial networks (GANs) and multivariate imputation by chained equations (MICE) on data sets of genetic variants from the 1000 Genomes Project.

The code in this repository has been adapted from the article

> Nußberger J, Boesel F, Lenz S, Binder H, Hess M. Synthetic observations from deep generative models and binary omics data with limited sample size. Briefings in Bioinformatics. 2020. doi:10.1093/bib/bbaa226.

The original data is from https://mathgen.stats.ox.ac.uk/impute/1000GP_Phase3.html.
The preprocessed data for the experiment can be found in the [data](data) folder.
Running the Julia script [`comparison.jl`](comparison.jl) produces the file [`result.tsv`](result.tsv).
This took about 10 hours on a cluster of three machines with 8, 12, and 20 cores with clock speeds of about 3 GHz and at least 8 GB RAM per core.
The R script [`comparisonplot.r`](comparisonplot.r) can be used to produce plots from the results.

The Julia code runs with Julia version 1.5.
The used packages are defined via the files [`Manifest.toml`](Manifest.toml) and [`Project.toml`](Project.toml).

