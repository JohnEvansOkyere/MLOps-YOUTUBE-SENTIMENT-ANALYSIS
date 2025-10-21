# MLOps-YOUTUBE-SENTIMENT-ANALYSIS

# MLflow-Basic-Demo


## For Dagshub:

MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/MLflow-Basic-Demo.mlflow \
MLFLOW_TRACKING_USERNAME=entbappy \
MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0 \
python script.py



```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/MLflow-Basic-Demo.mlflow

export MLFLOW_TRACKING_USERNAME=entbappy 

export MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0


```


# MLflow on AWS

## MLflow on AWS Setup:

1. Login to AWS console.
2. Create IAM user with AdministratorAccess
3. Export the credentials in your AWS CLI by running "aws configure"
4. Create a s3 bucket
5. Create EC2 machine (Ubuntu) & add Security groups 5000 port

Run the following command on EC2 machine
```bash
sudo apt update

sudo apt install python3-pip

sudo apt install pipenv

sudo apt install virtualenv

mkdir mlflow

cd mlflow

pipenv install mlflow

pipenv install awscli

pipenv install boto3

pipenv shell


## Then set aws credentials
aws configure


#Finally 
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflow-test-23 - replac mlflow-test-23 with your s3 bucket created

# Setting the Port number
Go to ec2, click on instance, click on security, selecet secuirty groups - Edit inbound rules - add rule - add your port number
#open Public IPv4 DNS to the port 5000


#set uri in your local terminal and in your code 
export MLFLOW_TRACKING_URI=http://ec2-54-147-36-34.compute-1.amazonaws.com:5000/
```


## DVC

dvc init

dvc repro

dvc dag



Before the pipelines
create the setup.py, yaml files
Start with Data ingestion - preprocessing - modeling - evaluation

But before model evaluation, download and install aws cli
run aws configure in the project directory on terminal
fill the requirements

after model evaluation, evaluations are saved inside the s3 bucket 
then the model registry use the run id generated to save the model to models in mlflow
run dvc repro to run the pipelines

I then built fastAPI to handle the backend, load model from the mlflow model registry
donwload postman and send a request to try it

create an chrome plugin extension, connect to your youtube API
load the extension in your chrome extensions, open a you

Now, we deploy it using Docker container

lets then create a .github/workflows/cicd.yaml for CI/CD automations


AWS CICD Deployment with Github Actions

2. Create IAM user for deployment

# Description About the deployment
1. BUild docer image of the source code

2. push your docker image to ECR

3. Launch your EC2

4. Pull your image from ECR in EC2

5. Launch your deocker image in EC2

#Policy:

1. AmazonEC2ContainerREgistryFullAccess

2. AmazonEC2FullAccess

## 3. Create ECR repo to store/save docker image

    - Save the URI: 699664936905.dkr.ecr.eu-north-1.amazonaws.com/mlproject

## 4. Create EC2 machine (Ubuntu)

## 5 Open EC2 and install docker in EC2 Machine

    #optional

    sudo apt-get update -y

    sudo apt-get upgrade

    #requirement

    curl -fsSl https://get.docker.com -o get-docker.sh

    sudo sh get-docker.sh

    sudo usermod -aG docker ubuntu
    
    docker --version, check if it is working

Open GitHub Project - Settings -Actions - Runner -New Self Host - Linux

    Copy and paste the commands in your EC2 virtual machine for configuration

    Enter name of runner - self-hosted
## 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one

    Now let's add some security credentials
    On Github - click Secrets and variables - AActions - repo-secrets

## 7. Setup github secrets
    from the EC2 secrets we downloaded in CSV

    AWS_ACCESS_KEY_ID = 

    AWS_SECRET_ACCESS_KEY = 

    AWS_REGION = eu-north-1

    AWS_ECR_LOGIN_URI = demo >> 699664936905.dkr.ecr.eu-north-1.amazonaws.com/mlflow-ecr

    ECR_REPOSITORY_NAME =mlflow-ecr 

