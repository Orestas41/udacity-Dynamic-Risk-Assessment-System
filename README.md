<h1 align='center'>A Dynamic Risk 
Assessment System</h1>
<h2 align='center'>ML DevOps Engineer Nanodegree</h2>
<h4 align='center'>Udacity</h4>
<h3 align='center'>Orestas Dulinskas</h3>
<h4 align='center'>November 2022</h4>

## Description
This project makes predictions about attrition risk in a fabricated dataset. It begins by setting up processes to ingest data and score, retrains and re-deploys ML models that predict attrition risk while writing scripts that automatically check for new data and model drift. APIs are set-up that allow users to access model results, metrics and diagnostics.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development purposes.

# Environment Set up
Download and install conda if you don’t have it already.
* conda create -n [envname] "python=3.8"
* pip install requirements.txt

# Running the full process

To run the full process:
* python fullprocess.py

# Output

The process generates plots and reports to monitor model performance:
* confusionmatrix2.png - Confusion Matrix
* apireturns2.txt:
    * Predictions from current dataset
    * Model score
    * Data scatistics:
        * Mean
        * Median
        * Standard Deviation
        * Percentage of NA files
    * Process diagnostics:
        * Timing of data ingestion and training processes
        * List of dependencies with current and latest version number