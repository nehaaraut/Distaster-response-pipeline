# Distaster Response Pipeline
This project is part of Data Science Nanodegree program by Udacity.

## Introduction
In this project, the task is to analyze disaster data from a set of messages to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events provided by Figure Eight. A machine learning pipeline is created to categorize these events so that one can send the messages to an appropriate disaster relief agency.

## Objectives
The data set used in this project contains real messages that were sent during disaster events. This project builds a machine learning pipeline to categorize these events so that the messages could be send to an appropriate disaster relief agency.

The Project is divided in the following Sections:

1. ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
2. Machine Learning Pipeline to train a model able to classify text messages in categories
3. Web App to show model results in real time.

## Prerequisites
1. Update python(anaconda) `$ conda update python`
2. Ctreate virtual envirnment(for windows).
`$ python -m venv disaster_project`
`$ disaster_project\Scripts\activate`
3. Install dependencies.
`$ pip install -r requirements.txt`

#### Dependencies

	click==7.1.2
	Flask==1.1.2
	itsdangerous==1.1.0
	Jinja2==2.11.2
	joblib==0.16.0
	MarkupSafe==1.1.1
	nltk==3.5
	numpy==1.19.1
	pandas==1.1.1
	plotly==4.9.0
	python-dateutil==2.8.1
	pytz==2020.1
	regex==2020.7.14
	retrying==1.3.3
	scikit-learn==0.23.2
	scipy==1.5.2
	six==1.15.0
	SQLAlchemy==1.3.19
	threadpoolctl==2.1.0
	tqdm==4.48.2
	Werkzeug==1.0.1

4. Following coomand can also be used to update all libraries in python.
`$ python -m pip install --upgrade pip`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/messages_categories.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/messages_categories.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:8000/

## File Structure

* `requirements.txt`: contains the environment requirements to run program
* `app` folder contains the following:
  * `templates`: Folder containing
    * `index.html`: Renders homepage
    * `go.html`: Renders the message classifier
  * `run.py`: Defines the app routes
* `data` folder contains the following:
    * `disaster_categories.csv`: contains the disaster categories csv file
    * `disaster_messages.csv`: contains the disaster messages csv file
    * `messages_categories.db`: contains the emergency db which merges categories and messages by ID
    * `process_data.py`: contains the scripts to transform data(ETL Pipeline)
* `models` folder contains the following:
    * `classifier.pkl`: contains the RandomForestClassifier pickle file
    * `train_classifier.py`: script to train ML_Pipeline.py



