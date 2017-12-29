# Fire Emissions AI

Fire Emissions AI is an [artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network) designed to predict fire emissions information based on previously collected data for an area. Fire Emmissions AI is trained using freely available data provided by the NASA EarthData [Global Fire Emissions Database V4.1](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1293)<sup>1</sup>.

# Quick Start Guide

1. You will need [Python 3.6](https://www.python.org/downloads/) or greater installed.
2. Install Pipenv, if you do not already have it, with `$ pip install --user pipenv`.
3. Run the project with `$ pipenv run python predictor/predict.py`.

The repository contains two main folders; `predictor` and `preprocessor`. The former contains the code relating to the neural network directly (including training, testing, and eventually predicting), the latter contains the code for a utility used to convert the Global Fire Emissions Database files into training examples for the predictor.

# References

1. Randerson, J.T., G.R. van der Werf, L. Giglio, G.J. Collatz, and P.S. Kasibhatla. 2017. Global Fire Emissions Database, Version 4.1 (GFEDv4). ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/1293
