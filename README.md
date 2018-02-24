# Fire Emissions AI

Fire Emissions AI is an [artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network) designed to predict fire emissions information based on previously collected data for an area. Fire Emmissions AI is trained using freely available data provided by the NASA EarthData [Global Fire Emissions Database (GFED) V4.1](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1293)<sup>1</sup>.

# Quick Start Guide

First you will need a `.csv` file with lines in the following format. Each value should be extracted directly from the GFED to ensure the correct format of individual values.

```
[Year],[Month],[Latitude],[Longitude],[Region ID],[BB],[NPP],[Rh],[C],[DM],[Burned Area]
```

1. You will need [Python 3.5](https://www.python.org/downloads/) or greater installed.
2. Install Pipenv, if you do not already have it, with `$ pip install pipenv`.
3. Install dependencies for the project using `$ pipenv install`
4. Run the unit tests and linters to ensure correct behavior `$ pipenv run python setup.py test`.
5. Run the predict.py with `$ pipenv run python fireemissionsai/predict.py [path to .csv]`.

The repository contains two main scripts; `fireemissionsai/predict.py` and `fireemissionsai/preprocess.py`. The former contains the code relating to the neural network directly (including training, testing, and predicting), the latter contains the code for a utility used to convert the Global Fire Emissions Database files into training examples for the predictor. The preprocess.py outputs `.csv` files that are used for training the predictor.

To train the predictor on your own data simply do the following.

1. Run the preprocess.py with `$ pipenv run python fireemissionsai/preprocess.py [GFED hdfs directory]`.
2. Run the predict.py with `$ pipenv run python fireemissionsai/predict.py [path to .csv] --retrain --persist`.

Using the `--persist` flag will save your newly trained model and weights over the existing model. To train a model for one-time-use simply omit the `--persist` flag.


# References

1. Randerson, J.T., G.R. van der Werf, L. Giglio, G.J. Collatz, and P.S. Kasibhatla. 2017. Global Fire Emissions Database, Version 4.1 (GFEDv4). ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/1293
