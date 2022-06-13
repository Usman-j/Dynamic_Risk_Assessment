# Dynamic Risk Assessment System
The fourth project for [ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) by Udacity.

## Description
The goal of the project is to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's clients. This is achieved by setting up automated processes to re-train, re-deploy and report on the ML model. The model is deployed via Flask API and the full pipeline is automated via cron job.


## Dependencies
Project dependencies are available in the [requirements](requirements.txt) file.

```bash
pip install -r requirements.txt
```
<img src="fullprocess.jpg" width=550 height=300>

## Detailed Steps
1. **Configuration:** A [configuration](config.json) file is used to set the paths of folders required for each step of the model development process.
2. **Data Ingestion:** [Ingestion](ingestion.py) script checks for datasets, compiles them together, writes to a merged [output file](/ingesteddata/finaldata.csv) and saves [record](/ingesteddata/ingestedfiles.txt) of ingested data.  
3. **Training:** [Training](training.py) script trains the model on the ingested data and saves the [pickled model](/practicemodels/trainedmodel.pkl).
4. **Scoring:** [Scoring](scoring.py) script takes a trained model, loads input data, calculates an F1 score and stores it in a [text file](/practicemodels//latestscore.txt).
5. **Deployment:** [Deployment](deployment.py) script copies the latest pickled model, the latestscore.txt file, and the ingestedfiles.txt file into the deployment directory as specified by the config file.
6. **Diagnostics:** [Diagnostics](diagnostics.py) script has multiple functions to determine and save summary statistics related to a dataset, time the performance of some other functions and check for dependency changes and package updates.
7. **Reporting:** [Reporting](reporting.py) script calculates a confusion matrix using the test data and the deployed model and saves the [figure](/practicemodels/confusionmatrix.png).
8. **App:** [App](app.py) script contains a Flask app to provide endpoints for various methods such as model predictions, scoring, diagnosis and data statistics.
9. **Process Automation:** All the previous steps are combined in a conditional flow in a [single script](fullprocess.py) such that model is re-trained and re-deployed only if new data with a model drift is detected. This script is run automatically at regular intervals with the help of a [cron job](cronjob.txt). 




