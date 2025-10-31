# MLOps-YOUTUBE-SENTIMENT-ANALYSIS

# MLflow-Basic-Demo


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
mlflow server --host 0.0.0.0 --default-artifact-root s3://youtube-sentiment00223 --allowed-hosts "*"    -   replac mlflow-test-23 with your s3 bucket created

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

3. AmazonS3FullAccess 
OR 
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::youtube-sentiment00223",
        "arn:aws:s3:::youtube-sentiment00223/*"
      ]
    }
  ]
}

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

    AWS_ECR_LOGIN_URI = demo >> 699664936905.dkr.ecr.eu-north-1.amazonaws.com/youtube-sentiment

    ECR_REPOSITORY_NAME =youtube-sentiment


    chrome://extensions

  # Backend Deployment

  build docker image on local machine

  test if it was working

  createed ecr repo on aws

  login to ecr

  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 699664936905.dkr.ecr.us-east-1.amazonaws.com 

Tag image
docker tag yt-sentiment-api:latest 699664936905.dkr.ecr.us-east-1.amazonaws.com/youtube-sentiment:latest

push ot ECR

docker push 699664936905.dkr.ecr.us-east-1.amazonaws.com/youtube-sentiment:latest

now, connecting or moving from ECR to EC2

RUN THIS TO MOVE THE SSH KEY TO SECURE PLACE
# Move to SSH directory
mv ~/Downloads/youtube-sentiment-current.pem ~/.ssh/

# Secure permissions (must be 400 or SSH fails)
chmod 400 ~/.ssh/youtube-sentiment-current.pem

# Verify
ls -la ~/.ssh/youtube-sentiment-current.pem  # Should show -r--------

run this
ssh -i ~/.ssh/youtube-sentiment-current.pem ubuntu@ec2-184-72-211-47.compute-1.amazonaws.com

install docker on ec2 machine
sudo apt update && sudo apt install -y 
docker.io && sudo usermod -aG 
docker ubuntu && newgrp docker

ON LOCAL MACHINE, transfer .env to ec2
scp -i ~/.ssh/youtube-sentiment-current.pem .env ubuntu@ec2-184-72-211-47.compute-1.amazonaws.com:~/

Login to ECR on EC2 machine
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 699664936905.dkr.ecr.us-east-1.amazonaws.com

login suceed

pull the image
docker pull 699664936905.dkr.ecr.us-east-1.amazonaws.com/youtube-sentiment:latest
