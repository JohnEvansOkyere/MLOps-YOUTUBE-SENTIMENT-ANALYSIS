# MLOps YouTube Sentiment Analysis

This project demonstrates a full **MLOps workflow** for sentiment analysis on YouTube comments, including **data pipelines, model training, evaluation, MLflow tracking, DVC pipelines, FastAPI backend, Chrome extension, Docker deployment, and AWS CI/CD setup**.

---

## Project Structure

```
MLOps-YOUTUBE-SENTIMENT-ANALYSIS/
├── data/                          # Data directory
│   ├── raw/                       # Raw data files
│   │   ├── test.csv
│   │   └── train.csv
│   └── interim/                   # Processed data files
│       ├── test_processed.csv
│       └── train_processed.csv
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── data/                      # Data processing modules
│   │   ├── data_ingestion.py
│   │   └── data_preprocessing.py
│   ├── model/                     # Model-related modules
│   │   ├── model_building.py
│   │   ├── model_evaluation.py
│   │   └── register_model.py
│   └── monitoring/                # Model monitoring modules
│       ├── __init__
│       ├── config/                 # Monitoring configuration
│       │   └── monitoring_config.yaml
│       ├── data_drift_detector.py
│       ├── model_monitor.py
│       └── report_generator.py
│
├── fastAPI_app/                   # FastAPI backend application
│   ├── __pycache__/
│   ├── main.py                    # Main FastAPI application
│   └── test.py                    # API tests
│
├── notebooks/                     # Jupyter notebooks for experimentation
│   ├── 1_Preprocessing_&_EDA.ipynb
│   ├── 2_experiment_1_baseline_model.ipynb
│   ├── 3_experiment_2_bow_.ipynb
│   ├── 4_experiment_3_Trigrams_(1,2)_max_features.ipynb
│   ├── 5_experiment_4_handling_imbalanced_data.ipynb
│   ├── 6_experiment_5_xgboost_with_hpt.ipynb
│   ├── 7_experiment_6_lightgbm_detailed_hpt.ipynb
│   ├── 8_stacking.ipynb
│   ├── confusion_matrix_adasyn.png
│   ├── confusion_matrix_class_weights.png
│   ├── confusion_matrix_oversampling.png
│   ├── confusion_matrix_smote_enn.png
│   ├── confusion_matrix_undersampling.png
│   ├── confusion_matrix.png
│   ├── dataset.csv
│   └── reddit_preprocessing.csv
│
├── yt-chrome-plugin-frontend/     # Chrome extension frontend
│   ├── manifest.json              # Extension manifest
│   ├── popup.html                  # Extension popup HTML
│   └── popup.js                    # Extension popup JavaScript
│
├── youtube.egg-info/              # Python package metadata
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   └── top_level.txt
│
├── Dockerfile                     # Docker container configuration
├── dvc.lock                       # DVC lock file for pipeline dependencies
├── dvc.yaml                       # DVC pipeline configuration
├── params.yaml                    # Model and pipeline parameters
├── requirements.txt               # Python dependencies
├── setup.py                       # Python package setup file
├── LICENSE                        # Project license
├── SECRET_READE.md               # Secrets/credentials documentation
├── README.md                      # This file
│
├── lgbm_model.pkl                # Saved LightGBM model file
├── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
├── confusion_matrix.png          # Confusion matrix visualization
├── confusion_matrix_Test Data.png # Test data confusion matrix
├── experiment_info.json          # Experiment metadata
├── errors.log                    # General error log
├── preprocessing_errors.log      # Preprocessing error log
├── model_building_errors.log     # Model building error log
├── model_evaluation_errors.log   # Model evaluation error log
└── model_registration_errors.log # Model registration error log
```
# **YouTube Sentiment Analysis - Production MLOps Pipeline**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-blue)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-latest-blue)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-Deployed-orange)](https://aws.amazon.com/)

> **Real-time sentiment analysis for YouTube comments with complete MLOps infrastructure - from data ingestion to production monitoring.**

---

## **📋 Table of Contents**

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Critical Problems Solved](#critical-problems-solved)
- [Prerequisites](#prerequisites)
- [Complete Setup Guide](#complete-setup-guide)
  - [1. Local Development Setup](#1-local-development-setup)
  - [2. AWS Infrastructure Setup](#2-aws-infrastructure-setup)
  - [3. MLflow Server Setup](#3-mlflow-server-setup)
  - [4. DVC Configuration](#4-dvc-configuration)
  - [5. Model Training Pipeline](#5-model-training-pipeline)
  - [6. Model Registration](#6-model-registration)
  - [7. FastAPI Deployment](#7-fastapi-deployment)
  - [8. CI/CD Pipeline Setup](#8-cicd-pipeline-setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## **🎯 Overview**

A production-ready machine learning system that analyzes sentiment in YouTube comments in real-time. This project demonstrates end-to-end MLOps practices including:

- ✅ **Reproducible ML pipelines** with DVC
- ✅ **Experiment tracking** with MLflow
- ✅ **Model versioning** and registry
- ✅ **Data drift detection** with Evidently AI
- ✅ **Automated CI/CD** with GitHub Actions
- ✅ **Production monitoring** and logging
- ✅ **Containerized deployment** with Docker
- ✅ **Cloud infrastructure** on AWS

**Live Demo:** [API Documentation](http://your-ec2-instance:8080/docs)

---

## **🏗️ Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Raw Data → Preprocessing → Feature Engineering → Training      │
│     ↓            ↓                ↓                    ↓          │
│   DVC          DVC             DVC                  MLflow       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      MODEL LIFECYCLE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Training → Evaluation → Drift Check → Registration → Deploy    │
│     ↓          ↓            ↓              ↓            ↓         │
│  MLflow    Evidently    Evidently    Model Registry   FastAPI   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      PRODUCTION SERVING                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Chrome Extension → FastAPI → Model → Monitoring → Alerting     │
│         ↓              ↓         ↓          ↓            ↓        │
│    User Request    Docker    MLflow   Evidently     Logs        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         CI/CD PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Git Push → Tests → Build → ECR → Deploy → Health Check         │
│      ↓        ↓       ↓      ↓       ↓          ↓                │
│  GitHub   Actions  Docker   AWS    EC2      Verify              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## **🛠️ Tech Stack**

### **Machine Learning**
- **LightGBM** - Gradient boosting framework for classification
- **scikit-learn** - Feature engineering (TF-IDF vectorization)
- **NLTK** - Text preprocessing (lemmatization, stopwords)
- **pandas** - Data manipulation

### **MLOps Infrastructure**
- **MLflow** - Experiment tracking, model registry, artifact storage
- **DVC** - Data versioning, pipeline orchestration
- **Evidently AI** - Data drift detection, model monitoring
- **Git** - Version control

### **Backend & API**
- **FastAPI** - High-performance web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### **DevOps & Deployment**
- **Docker** - Containerization
- **AWS EC2** - Compute instance
- **AWS S3** - Artifact storage
- **AWS ECR** - Container registry
- **GitHub Actions** - CI/CD automation

### **Monitoring & Logging**
- **Python logging** - Application logs
- **Evidently Reports** - Model performance monitoring
- **MLflow Tracking** - Experiment metrics

---

## **🔥 Critical Problems Solved**

### **1. Data Quality Issues**
**Problem:** Production data had 57 duplicates and 26 missing values causing model degradation.

**Solution:**
- Implemented comprehensive data validation in preprocessing pipeline
- Added quality checks with Evidently AI before training
- Automated duplicate detection and removal
- Missing value handling with proper imputation

**Code:** `src/data/data_preprocessing.py` - `clean_data()` function

---

### **2. Model Drift Detection**
**Problem:** Model accuracy dropped from 85% to 62% in production without detection.

**Solution:**
- Integrated Evidently AI for automated drift detection
- Reference data tracking for distribution comparison
- Automated alerts when drift is detected
- Monthly drift reports saved to `reports/drift/`

**Code:** `src/monitoring/data_drift_detector.py`

---

### **3. Experiment Tracking Confusion**
**Problem:** Training and evaluation both created `experiment_info.json`, overwriting model registration data.

**Solution:**
- Training creates experiment_info.json with model metadata
- Evaluation appends results without overwriting
- Separate run_ids for training vs evaluation
- Clear separation in DVC pipeline

**Code:** `src/model/model_building.py` - `save_experiment_info()`  
**Code:** `src/model/model_evaluation.py` - `save_evaluation_info()`

---

### **4. Model Versioning & Rollback**
**Problem:** No way to track which model version was in production or rollback bad deployments.

**Solution:**
- MLflow Model Registry with staging/production stages
- Model aliasing system (champion/challenger)
- Automated rollback in CI/CD pipeline
- Version tracking in all predictions

**Code:** `src/model/register_model.py`

---

### **5. Preprocessing Inconsistency**
**Problem:** Training and inference used different preprocessing, causing prediction errors.

**Solution:**
- Single source of truth for preprocessing
- FastAPI imports from training pipeline
- Shared `preprocess_comment()` function
- Identical NLTK settings across environments

**Code:** `src/data/data_preprocessing.py` (used by both training and FastAPI)

---

### **6. S3 Model Artifact Access**
**Problem:** Model artifacts in S3 couldn't be accessed by registration script (404 errors).

**Solution:**
- Proper AWS credentials configuration
- MLflow S3 endpoint URL setup
- Fallback to local files if MLflow unavailable
- Validation before registration

**Code:** `fastAPI_app/main.py` - `load_model_from_mlflow()` with fallback

---

### **7. Zero-Downtime Deployment**
**Problem:** Stopping old container before starting new one caused service interruption.

**Solution:**
- Health checks before stopping old container
- Automated rollback on deployment failure
- Container naming strategy for clean transitions
- Post-deployment verification tests

**Code:** `.github/workflows/main.yml` - deployment job

---

## **📦 Prerequisites**

### **Required Accounts**
- [x] GitHub account
- [x] AWS account (with billing enabled)
- [x] YouTube Data API key ([Get it here](https://console.developers.google.com))

### **Local Machine Requirements**
```bash
# Software
- Python 3.11+
- Git
- Docker & Docker Compose
- AWS CLI v2

# Operating System
- Ubuntu 20.04+ / macOS / Windows with WSL2
```

### **AWS Resources Needed**
- EC2 instance (t2.medium or larger) - for MLflow server
- EC2 instance (t2.small or larger) - for production API (or same as above)
- S3 bucket - for artifacts and data storage
- ECR repository - for Docker images
- IAM user with appropriate permissions

---

## **🚀 Complete Setup Guide**

### **1. Local Development Setup**

#### **Step 1.1: Clone Repository**
```bash
# Clone the repository
git clone https://github.com/yourusername/MLOps-YOUTUBE-SENTIMENT-ANALYSIS.git
cd MLOps-YOUTUBE-SENTIMENT-ANALYSIS

# Create virtual environment
python3.11 -m venv youtube
source youtube/bin/activate  # On Windows: youtube\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### **Step 1.2: Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Verify installation
python -c "import lightgbm, mlflow, dvc, evidently, fastapi; print('✓ All packages installed')"
```

#### **Step 1.3: Configure Environment Variables**
```bash
# Create .env file
cp .env.example .env

# Edit .env with your values
nano .env
```

**Required `.env` contents:**
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=youtube-sentiment-mlops

# MLflow Configuration
MLFLOW_TRACKING_URI=http://your-ec2-ip:5000
MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com

# Model Configuration
MODEL_REGISTRY_NAME=yt_chrome_plugin_model
MODEL_ALIAS=champion

# YouTube API
YOUTUBE_API_KEY=your_youtube_api_key

# Application Settings
DEBUG=false
ENABLE_MONITORING=true
MONITORING_SAMPLE_RATE=0.1
```

#### **Step 1.4: Initialize Git and DVC**
```bash
# Initialize Git (if needed)
git init
git add .
git commit -m "Initial commit"

# Initialize DVC
dvc init

# Configure DVC remote (S3)
dvc remote add -d myremote s3://youtube-sentiment-mlops/dvc-storage
dvc remote modify myremote region us-east-1

# Verify DVC configuration
dvc remote list
dvc status
```

---

### **2. AWS Infrastructure Setup**

#### **Step 2.1: Create S3 Bucket**
```bash
# Create S3 bucket for artifacts
aws s3 mb s3://youtube-sentiment-mlops --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket youtube-sentiment-mlops \
  --versioning-configuration Status=Enabled

# Verify
aws s3 ls
```

#### **Step 2.2: Create ECR Repository**
```bash
# Create ECR repository
aws ecr create-repository \
  --repository-name yt-sentiment-api \
  --region us-east-1

# Get repository URI (save this)
aws ecr describe-repositories \
  --repository-names yt-sentiment-api \
  --query 'repositories[0].repositoryUri' \
  --output text
```

#### **Step 2.3: Launch EC2 Instance for MLflow**
```bash
# Launch EC2 instance (t2.medium recommended)
# - AMI: Ubuntu 22.04 LTS
# - Instance Type: t2.medium
# - Storage: 30 GB gp3
# - Security Group: Open ports 5000, 22

# Save the instance public IP for later
```

**Security Group Rules:**
```
Inbound Rules:
- Port 22 (SSH) - Your IP
- Port 5000 (MLflow) - 0.0.0.0/0 (or restrict to your IP)
- Port 8080 (FastAPI) - 0.0.0.0/0 (if using same instance)
```

#### **Step 2.4: Create IAM User for GitHub Actions**
```bash
# Create IAM policy (save as github-actions-policy.json)
cat > github-actions-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::youtube-sentiment-mlops",
        "arn:aws:s3:::youtube-sentiment-mlops/*"
      ]
    }
  ]
}
EOF

# Create IAM user
aws iam create-user --user-name github-actions-user

# Attach policy
aws iam create-policy \
  --policy-name GitHubActionsPolicy \
  --policy-document file://github-actions-policy.json

aws iam attach-user-policy \
  --user-name github-actions-user \
  --policy-arn arn:aws:iam::YOUR_ACCOUNT_ID:policy/GitHubActionsPolicy

# Create access keys (save these securely!)
aws iam create-access-key --user-name github-actions-user
```

---

### **3. MLflow Server Setup**

#### **Step 3.1: SSH into EC2 Instance**
```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# Update system
sudo apt update && sudo apt upgrade -y
```

#### **Step 3.2: Install MLflow Dependencies**
```bash
# Install Python and pip
sudo apt install -y python3.11 python3.11-venv python3-pip

# Create mlflow user
sudo useradd -m -s /bin/bash mlflow

# Create directories
sudo mkdir -p /opt/mlflow
sudo chown mlflow:mlflow /opt/mlflow
```

#### **Step 3.3: Setup MLflow**
```bash
# Switch to mlflow user
sudo su - mlflow

# Create virtual environment
python3.11 -m venv /opt/mlflow/venv
source /opt/mlflow/venv/bin/activate

# Install MLflow and dependencies
pip install mlflow boto3 psycopg2-binary

# Configure AWS credentials
mkdir -p ~/.aws
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
region = us-east-1
EOF

cat > ~/.aws/config << EOF
[default]
region = us-east-1
output = json
EOF
```

#### **Step 3.4: Create MLflow Systemd Service**
```bash
# Exit mlflow user
exit

# Create service file
sudo nano /etc/systemd/system/mlflow.service
```

**Paste this content:**
```ini
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=mlflow
WorkingDirectory=/opt/mlflow
Environment="PATH=/opt/mlflow/venv/bin"
ExecStart=/opt/mlflow/venv/bin/mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///opt/mlflow/mlflow.db \
    --default-artifact-root s3://youtube-sentiment-mlops/mlflow-artifacts
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### **Step 3.5: Start MLflow Server**
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable and start MLflow
sudo systemctl enable mlflow
sudo systemctl start mlflow

# Check status
sudo systemctl status mlflow

# View logs
sudo journalctl -u mlflow -f

# Verify
curl http://localhost:5000/api/2.0/mlflow/experiments/list
```

**Test from your local machine:**
```bash
# Replace with your EC2 IP
curl http://your-ec2-ip:5000
# Should return MLflow UI HTML
```

---

### **4. DVC Configuration**

#### **Step 4.1: Configure DVC Remote**
```bash
# On your local machine
cd MLOps-YOUTUBE-SENTIMENT-ANALYSIS

# Add S3 remote
dvc remote add -d myremote s3://youtube-sentiment-mlops/dvc-storage
dvc remote modify myremote region us-east-1

# Configure AWS credentials for DVC
dvc remote modify myremote access_key_id YOUR_ACCESS_KEY
dvc remote modify myremote secret_access_key YOUR_SECRET_KEY

# Or use AWS CLI credentials
dvc remote modify myremote profile default

# Commit DVC config
git add .dvc/config
git commit -m "Configure DVC remote"
```

#### **Step 4.2: Verify DVC Setup**
```bash
# Check DVC status
dvc status

# Test DVC remote
echo "test" > test.txt
dvc add test.txt
dvc push

# Verify in S3
aws s3 ls s3://youtube-sentiment-mlops/dvc-storage/ --recursive

# Clean up test
rm test.txt test.txt.dvc
git rm test.txt.dvc
```

---

### **5. Model Training Pipeline**

#### **Step 5.1: Prepare Data**
```bash
# Create data directories
mkdir -p data/raw data/interim

# Option 1: Download dataset (if you have a source)
# wget YOUR_DATASET_URL -O data/raw/youtube_comments.csv

# Option 2: Use the data ingestion script
python src/data/data_ingestion.py

# Verify data
ls -lh data/raw/
head data/raw/train.csv
```

#### **Step 5.2: Run Complete Pipeline**
```bash
# Option 1: Run entire pipeline with DVC
dvc repro

# Option 2: Run stages individually
dvc repro data_ingestion
dvc repro data_preprocessing
dvc repro reference_data_creation
dvc repro model_building
dvc repro model_evaluation

# View pipeline
dvc dag

# Check metrics
dvc metrics show

# View plots
dvc plots show
```

#### **Step 5.3: Verify Training Results**
```bash
# Check experiment info
cat experiment_info.json

# Verify model files exist
ls -lh *.pkl

# Check MLflow UI
# Open: http://your-ec2-ip:5000
# You should see your experiment with metrics

# Check reports
ls -lh reports/metrics/
ls -lh reports/drift/
ls -lh reports/data_quality/
```

#### **Step 5.4: Push to DVC Remote**
```bash
# Push data and model artifacts
dvc push

# Verify in S3
aws s3 ls s3://youtube-sentiment-mlops/dvc-storage/ --recursive

# Commit changes
git add dvc.lock experiment_info.json
git commit -m "Training run - Accuracy: 78.4%"
git push
```

---

### **6. Model Registration**

#### **Step 6.1: Register Model to MLflow**
```bash
# Make sure experiment_info.json exists and has correct run_id
cat experiment_info.json | jq '.run_id'

# Register model
python src/model/register_model.py

# Expected output:
# ✓ Model registered successfully
# ✓ Model transitioned to: Staging
# Model name: yt_chrome_plugin_model
# Version: 1
```

#### **Step 6.2: Verify in MLflow UI**
```bash
# Open MLflow UI
# Navigate to: http://your-ec2-ip:5000

# Steps to verify:
# 1. Click "Models" in top menu
# 2. Find "yt_chrome_plugin_model"
# 3. Check "Staging" stage has version 1
# 4. View model details and artifacts
```

#### **Step 6.3: Promote to Production**
```bash
# Option 1: Using registration script (with manual approval)
python src/model/register_model.py
# Follow prompts to promote to Production

# Option 2: Using MLflow CLI
mlflow models update \
  --name yt_chrome_plugin_model \
  --version 1 \
  --stage Production

# Option 3: Set alias (recommended)
mlflow models set-alias \
  --name yt_chrome_plugin_model \
  --alias champion \
  --version 1

# Verify
mlflow models list --name yt_chrome_plugin_model
```

---

### **7. FastAPI Deployment**

#### **Step 7.1: Test Locally First**
```bash
# Update .env with MLflow URI
echo "MLFLOW_TRACKING_URI=http://your-ec2-ip:5000" >> .env

# Run FastAPI locally
python fastAPI_app/main.py

# In another terminal, test endpoints
curl http://localhost:8080/health

curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"comments": ["This video is amazing!"]}'

# Check API docs
# Open: http://localhost:8080/docs
```

#### **Step 7.2: Build and Test Docker Image**
```bash
# Build Docker image
docker build -t yt-sentiment-api:latest .

# Run container locally
docker run -d \
  --name yt-sentiment-test \
  -p 8080:8080 \
  --env-file .env \
  yt-sentiment-api:latest

# Test container
curl http://localhost:8080/health

# Check logs
docker logs -f yt-sentiment-test

# Stop and remove
docker stop yt-sentiment-test
docker rm yt-sentiment-test
```

#### **Step 7.3: Push to ECR**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag yt-sentiment-api:latest \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/yt-sentiment-api:latest

# Push to ECR
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/yt-sentiment-api:latest

# Verify
aws ecr describe-images \
  --repository-name yt-sentiment-api \
  --region us-east-1
```

#### **Step 7.4: Deploy to EC2**

**Option A: Deploy on same EC2 as MLflow**
```bash
# SSH to EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
# Logout and login again

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Pull image
docker pull YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/yt-sentiment-api:latest

# Create .env file on EC2
nano .env
# Paste your environment variables

# Run container
docker run -d \
  --name yt-sentiment-api \
  --restart unless-stopped \
  -p 8080:8080 \
  --env-file .env \
  -v /var/log/yt-sentiment:/app/logs \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/yt-sentiment-api:latest

# Check logs
docker logs -f yt-sentiment-api

# Test
curl http://localhost:8080/health
```

**Option B: Deploy on separate EC2**
```bash
# Launch new t2.small EC2 instance
# - Same security group (port 8080 open)
# - Same key pair

# Follow same steps as Option A
```

#### **Step 7.5: Verify Production Deployment**
```bash
# From your local machine
# Test health endpoint
curl http://your-api-ec2-ip:8080/health

# Test prediction
curl -X POST http://your-api-ec2-ip:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "comments": [
      "This is amazing!",
      "I hate this",
      "Not sure about this"
    ]
  }'

# Access API documentation
# Open: http://your-api-ec2-ip:8080/docs

# Check monitoring logs
ssh -i your-key.pem ubuntu@your-api-ec2-ip
docker logs -f yt-sentiment-api
```

---

### **8. CI/CD Pipeline Setup**

#### **Step 8.1: Setup GitHub Secrets**
```bash
# Go to GitHub repository
# Settings → Secrets and variables → Actions → New repository secret

# Add these secrets:
```

**Required Secrets:**
| Secret Name | Value | Description |
|------------|-------|-------------|
| `AWS_ACCESS_KEY_ID` | `AKIA...` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | `secret123...` | IAM user secret key |
| `AWS_REGION` | `us-east-1` | AWS region |
| `ECR_REPOSITORY_NAME` | `yt-sentiment-api` | ECR repo name |
| `AWS_ECR_LOGIN_URI` | `123456.dkr.ecr.us-east-1.amazonaws.com` | ECR URI |
| `MLFLOW_TRACKING_URI` | `http://ec2-ip:5000` | MLflow server |
| `MODEL_REGISTRY_NAME` | `yt_chrome_plugin_model` | Model name |
| `MODEL_ALIAS` | `champion` | Model alias |
| `YOUTUBE_API_KEY` | `AIza...` | YouTube API key |

#### **Step 8.2: Setup Self-Hosted Runner**
```bash
# SSH to your production EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Navigate to GitHub repo → Settings → Actions → Runners
# Click "New self-hosted runner"
# Select Linux x64

# Follow the setup instructions (example):
mkdir actions-runner && cd actions-runner

# Download runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz \
  -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Configure (use token from GitHub)
./config.sh --url https://github.com/YOUR_USERNAME/YOUR_REPO \
  --token YOUR_TOKEN

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start

# Verify
sudo ./svc.sh status

# Check GitHub - runner should show as "Idle" (green dot)
```

#### **Step 8.3: Test CI/CD Pipeline**
```bash
# On your local machine

# Make a small change
echo "# Test CI/CD" >> README.md

# Commit and push
git add .
git commit -m "Test CI/CD pipeline"
git push origin main

# Watch GitHub Actions
# Go to: https://github.com/YOUR_USERNAME/YOUR_REPO/actions

# Monitor workflow:
# 1. Code quality checks should pass
# 2. Pipeline validation should pass
# 3. Docker build and push to ECR
# 4. Deployment to EC2
# 5. Health check verification

# Check deployment
curl http://your-ec2-ip:8080/health

# Verify in MLflow
# Open: http://your-ec2-ip:5000
```

#### **Step 8.4: Test Rollback Mechanism**
```bash
# Intentionally break the API to test rollback

# Create a branch
git checkout -b test-rollback

# Break something (e.g., make health endpoint fail)
# Edit fastAPI_app/main.py
sed -i 's/return {/return error {/g' fastAPI_app/main.py

# Commit and push
git add .
git commit -m "Test: Break health endpoint"
git push origin test-rollback

# Create pull request and merge to main
# Watch GitHub Actions

# Expected behavior:
# 1. Build succeeds
# 2. Deployment fails health check
# 3. Automatic rollback to previous version
# 4. Notification of failure

# Verify service is still running (old version)
curl http://your-ec2-ip:8080/health
# Should return healthy response (old version)

# Fix the issue
git checkout main
git revert HEAD
git push origin main

# Watch successful deployment
```

---

## **📁 Project Structure**

```
MLOps-YOUTUBE-SENTIMENT-ANALYSIS/
│
├── .github/
│   └── workflows/
│       └── main.yml                 # CI/CD pipeline
│
├── data/
│   ├── raw/                         # Raw data (not in Git)
│   └── interim/                     # Processed data (not in Git)
│
├── reference_data/
│   └── train_reference.csv          # Reference for drift detection
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py       # Download/split data
│   │   └── data_preprocessing.py   # Clean & preprocess
│   │
│   ├── model/
│   │   ├── model_building.py       # Train model with MLflow
│   │   ├── model_evaluation.py     # Evaluate & check drift
│   │   └── register_model.py       # Register to MLflow
│   │
│   └── monitoring/
│       └── data_drift_detector.py  # Drift detection logic
│
├── fastAPI_app/
│   └── main.py                      # Production API
│
├── reports/
│   ├── metrics/                     # Model metrics
│   ├── confusion_matrices/          # Confusion matrix plots
│   ├── drift/                       # Drift detection reports
│   └── data_quality/                # Data quality reports
│
├── logs/                            # Application logs
│
├── scripts/
│   └── create_reference_data.py    # Create reference dataset
│
├── notebooks/                       # Jupyter notebooks (EDA)
│
├── Dockerfile                       # Container definition
├── docker-compose.yml               # Local Docker setup
├── .dockerignore                    # Docker ignore rules
├── requirements.txt                 # Python dependencies
├── dvc.yaml                         # DVC pipeline definition
├── params.yaml                      # Model hyperparameters
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
├── .dvcignore                       # DVC ignore rules
├── experiment_info.json             # Latest training run info
├── lgbm_model.pkl                   # Trained model
├── tfidf_vectorizer.pkl            # Trained vectorizer
└── README.md                        # This file
```

---

## **💻 Usage**

### **Training a New Model**
```bash
# 1. Update hyperparameters (optional)
nano params.yaml

# 2. Run pipeline
dvc repro

# 3. Review results
dvc metrics show
mlflow ui  # Open http://localhost:5000

# 4. If satisfied, register model
python src/model/register_model.py

# 5. Push changes
dvc push
git add dvc.lock experiment_info.json
git commit -m "Training run v2"
git push
```

### **Making Predictions**

**Python:**
```python
import requests

response = requests.post(
    "http://your-api:8080/predict",
    json={
        "comments": [
            "This video is amazing!",
            "I don't like this content"
        ]
    }
)

predictions = response.json()
for pred in predictions:
    print(f"{pred['comment']}: {pred['sentiment']} (confidence: {pred['confidence']})")
```

**cURL:**
```bash
curl -X POST http://your-api:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "comments": [
      "Great tutorial!",
      "This is terrible"
    ]
  }'
```

**Chrome Extension:**
1. Install extension from `chrome-extension/` folder
2. Navigate to any YouTube video
3. Click extension icon
4. View real-time sentiment analysis

### **Checking Model Performance**
```bash
# View latest metrics
cat reports/metrics/metrics_test_latest.json

# Check drift reports
ls -lt reports/drift/

# View Evidently HTML report
open reports/drift/drift_report_*.html

# Check MLflow
# Open: http://your-mlflow-server:5000
# Navigate to your experiment
```

### **Switching Model Versions**
```bash
# List available versions
mlflow models list --name yt_chrome_plugin_model

# Set new champion
mlflow models set-alias \
  --name yt_chrome_plugin_model \
  --alias champion \
  --version 2

# API will automatically use new model on next reload
# Or trigger manual reload via API:
curl -X POST http://your-api:8080/model/reload \
  -d '{"alias": "champion"}'
```

---

## **📊 Monitoring & Maintenance**

### **Daily Checks**
```bash
# Check API health
curl http://your-api:8080/health

# Check container status
ssh your-ec2 "docker ps"

# Check recent logs
ssh your-ec2 "docker logs --tail 100 yt-sentiment-api"

# Check MLflow experiments
# Open: http://your-mlflow:5000
```

### **Weekly Checks**
```bash
# Review drift reports
ls -lt reports/drift/

# Check model performance trends
# Compare metrics across runs in MLflow

# Review inference logs
ssh your-ec2 "tail -n 1000 /var/log/yt-sentiment/*.jsonl"

# Check disk usage
ssh your-ec2 "df -h"

# Clean old Docker images
ssh your-ec2 "docker system prune -f"
```

### **Monthly Tasks**
```bash
# Retrain model with new data
dvc repro -f

# Compare with production model
# If new model better:
python src/model/register_model.py

# Update reference data for drift detection
python scripts/create_reference_data.py

# Backup MLflow database
ssh mlflow-server "cp /opt/mlflow/mlflow.db /opt/mlflow/backups/mlflow_$(date +%Y%m%d).db"
```

### **Alerts to Setup**

**CloudWatch Alarms (Recommended):**
```bash
# EC2 CPU > 80%
# EC2 Disk > 90%
# API 5xx errors > 10 per minute
# Container restarts
```

**Custom Alerts:**
```bash
# Drift detected → Slack/Email
# Model accuracy drops > 5% → Slack/Email
# API latency > 1s → Slack/Email
```

---

## **🔧 Troubleshooting**

### **Issue: Model not loading from MLflow**

**Symptoms:**
```
ERROR: Failed to load model from MLflow
```

**Solutions:**
```bash
# 1. Check MLflow server is running
curl http://your-mlflow:5000/health

# 2. Verify AWS credentials
aws s3 ls s3://your-bucket/

# 3. Check model exists in registry
mlflow models list --name yt_chrome_plugin_model

# 4. Try loading manually
python -c "
import mlflow
mlflow.set_tracking_uri('http://your-mlflow:5000')
model = mlflow.sklearn.load_model('models:/yt_chrome_plugin_model/1')
print('Model loaded!')
"

# 5. Check API logs
docker logs yt-sentiment-api | grep -i "model"
```

---

### **Issue: DVC push fails**

**Symptoms:**
```
ERROR: failed to push data to remote
```

**Solutions:**
```bash
# 1. Check AWS credentials
aws sts get-caller-identity

# 2. Verify S3 bucket access
aws s3 ls s3://your-bucket/

# 3. Check DVC remote config
dvc remote list
cat .dvc/config

# 4. Try manual push with verbose
dvc push -v

# 5. Reset DVC cache if corrupted
dvc cache clean
dvc pull
```

---

### **Issue: Drift detection failing**

**Symptoms:**
```
ERROR: Reference data not found
```

**Solutions:**
```bash
# 1. Create reference data
python scripts/create_reference_data.py

# 2. Verify reference data exists
ls -lh reference_data/train_reference.csv

# 3. Check data format
head reference_data/train_reference.csv

# 4. Re-run evaluation
dvc repro -f model_evaluation
```

---

### **Issue: CI/CD pipeline fails**

**Symptoms:**
- GitHub Actions shows red ❌
- Deployment doesn't complete

**Solutions:**
```bash
# 1. Check GitHub Actions logs
# Go to: repo → Actions → failed workflow → View logs

# 2. Verify GitHub secrets are set
# Settings → Secrets → Actions

# 3. Check self-hosted runner
ssh your-ec2 "cd actions-runner && ./run.sh"

# 4. Test Docker build locally
docker build -t test .

# 5. Check EC2 has space
ssh your-ec2 "df -h"

# 6. Manually pull and run to debug
ssh your-ec2
docker pull YOUR_ECR_URI/yt-sentiment-api:latest
docker run -it --rm YOUR_ECR_URI/yt-sentiment-api:latest bash
```

---

### **Issue: API returning errors**

**Symptoms:**
```
500 Internal Server Error
```

**Solutions:**
```bash
# 1. Check container logs
docker logs -f yt-sentiment-api

# 2. Check environment variables
docker exec yt-sentiment-api env | grep MLFLOW

# 3. Test health endpoint
curl http://localhost:8080/health

# 4. Restart container
docker restart yt-sentiment-api

# 5. Check if model loaded
docker logs yt-sentiment-api | grep "Model loaded"

# 6. Test with simple prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"comments": ["test"]}'
```

---

### **Issue: Out of disk space**

**Symptoms:**
```
No space left on device
```

**Solutions:**
```bash
# 1. Check disk usage
df -h

# 2. Clean Docker
docker system prune -a -f
docker volume prune -f

# 3. Clean old logs
sudo find /var/log -type f -name "*.log" -mtime +30 -delete

# 4. Clean DVC cache (if on local machine)
dvc gc -w

# 5. If on EC2, increase EBS volume:
# AWS Console → EC2 → Volumes → Modify → Increase size
# Then: sudo resize2fs /dev/xvda1
```

---

## **🤝 Contributing**

```bash
# 1. Fork the repository

# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Make changes and test
dvc repro
pytest tests/

# 4. Commit changes
git add .
git commit -m "Add amazing feature"

# 5. Push to branch
git push origin feature/amazing-feature

# 6. Create Pull Request
```

---

## **📝 License**

MIT License - feel free to use this project for learning and development.

---

## **📧 Contact**

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

**Project Link:** [https://github.com/yourusername/MLOps-YOUTUBE-SENTIMENT-ANALYSIS](https://github.com/yourusername/MLOps-YOUTUBE-SENTIMENT-ANALYSIS)

**LinkedIn:** [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

## **🙏 Acknowledgments**

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Evidently AI](https://docs.evidentlyai.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LightGBM](https://lightgbm.readthedocs.io/)

---

## **📚 Additional Resources**

- **Tutorial:** [Setting up MLflow on AWS](https://mlflow.org/docs/latest/tracking.html#amazon-s3)
- **Guide:** [DVC with S3](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3)
- **Best Practices:** [MLOps Principles](https://ml-ops.org/content/mlops-principles)
- **Monitoring:** [Evidently AI Blog](https://evidentlyai.com/blog)

---

<p align="center">
  Made with ❤️ for MLOps enthusiasts
</p>

<p align="center">
  ⭐ Star this repo if you found it helpful!
</p>

---

**Last Updated:** October 31, 2025