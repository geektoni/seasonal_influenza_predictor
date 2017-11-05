# Seasonal Influenza Predictor

## Introduction

This directory contains the code made for my bachelor's thesis. The main purpose
of this work was to test if we can predict influenza season in Italy by looking
at Wikipedia's page view count of specific pages, and thus provide a new, effective
way to estimate the influenza incidence among the population at a minimum cost and
in less time.

This project starting point was this [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3990502/), in which they try to do the same
considering USA influenza seasons and English Wikipedia's data. The results presented in that paper are
very promising and we tried to see if the same applies in other countries (Italy). 

## Repository structure

The repo was organized following this directory structure:
* `data`: it contains the actual data used to train and test the models. You can
find Wikipedia parsed page view logs, the list of Wikipedia's pages used and the
real influenza incidence data taken from official italian surveillance system
(InfluNet);
* `data_analysis`: it contains scripts used to parse and analyze Wikipedia page view
data. These script were used to generate the datasets inside the `data` directory;
* `models`: it contains the actual machine learning models (Linear and Poisson);
* `pagecounts`: it contains the code used to download and store the complete page
view logs.
