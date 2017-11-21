# Churn Prediction using AMLWorkbench - Modeling and Evaluation in Docker

## 1. Objectives

The aim of this lab is to generate churn classifiers in a docker container on the local computer. We can use the .dprep file created in the previous labs. However, in this lab, we will reproduce the steps related to data preparation using pandas for flexibility.

## 2. Data Preparation

The csv file can be read using pandas into a dataframe df. The data preparation tasks such as dropping columns and removing duplicates can be performed using drop and drop_duplicates functions from pandas:

```
df = pd.read_csv('CATelcoCustomerChurnTrainingSample.csv')
df = df.drop('year', 1)
df = df.drop('month', 1)
df = df.drop_duplicates()
```
Rest of the code related to one-hot encoding, splitting the data and modeling is pretty much the same as in the previous lab. Ensure that the CATelcoCustomerChurnTrainingSample.csv is in the root folder where the code is.

## 3. Execution – Local Docker Container

Ensure conda_dependencies.yml contains the following dependencies:
python=3.5.2 and scikit-learn

If you have a Docker engine running locally, in the CLI window, run the below command. Note the change the run configuration from local to docker. PrepareEnvironment must be set to true in aml_config/docker.runconfig before you can submit.

```
az ml experiment submit -c docker CATelcoCustomerChurnModelingDocker.py
```

This command pulls down a base docker image, layers a conda environment on the base image based on the conda_dependencies.yml file in your_aml_config_ directory, and then starts a Docker container. It then executes your script. You should see some Docker image construction messages in the CLI window. In the end, on successful execution, you will see result as shown below.

![DockerEngine](https://github.com/Azure/MachineLearningSamples-ChurnPrediction/blob/master/docs/Images/DockerEngine.png)

You will see different accuracy measures this time compared to the previous lab. This is because of the random train and test datasets in train_test_split.


[Go to next hands-on lab](https://github.com/Azure/MachineLearningSamples-ChurnPrediction/blob/master/docs/Operationalization.md)
