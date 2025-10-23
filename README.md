Sure! Here's a **ready-to-copy README** for your project:

````markdown
# MLOps YouTube Sentiment Analysis

This project demonstrates a full **MLOps workflow** for sentiment analysis on YouTube comments, including **data pipelines, model training, evaluation, MLflow tracking, DVC pipelines, FastAPI backend, Chrome extension, Docker deployment, and AWS CI/CD setup**.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Setup & Dependencies](#setup--dependencies)  
3. [MLflow Tracking](#mlflow-tracking)  
   - [Dagshub](#dagshub)  
   - [AWS MLflow Server](#aws-mlflow-server)  
4. [DVC Pipelines](#dvc-pipelines)  
5. [Backend & API](#backend--api)  
6. [Chrome Extension](#chrome-extension)  
7. [Docker Deployment](#docker-deployment)  
8. [AWS CI/CD with GitHub Actions](#aws-cicd-with-github-actions)  

---

## Project Overview

The project performs **YouTube comment sentiment analysis** using the following components:  

- **Data Pipelines**: Data ingestion → Preprocessing → Modeling → Evaluation → Model Registry.  
- **MLflow**: Tracks experiments, stores artifacts, and registers models.  
- **DVC**: Version control for data and reproducible pipelines.  
- **FastAPI**: Backend to serve predictions and charts.  
- **Chrome Extension**: UI to analyze YouTube comments directly in the browser.  
- **Docker & AWS**: Containerized deployment with CI/CD using GitHub Actions.  

---

## Setup & Dependencies

1. Clone the repository:

```bash
git clone https://github.com/JohnEvansOkyere/MLOps-YOUTUBE-SENTIMENT-ANALYSIS.git
cd MLOps-YOUTUBE-SENTIMENT-ANALYSIS
````

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Optional: Setup virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```


---

### AWS MLflow Server

1. **Create AWS Resources**:

   * IAM user with `AdministratorAccess`
   * S3 bucket for artifacts
   * EC2 instance (Ubuntu) with port 5000 open in Security Group

2. **Setup EC2**:

```bash
sudo apt update
sudo apt install python3-pip -y
sudo apt install pipenv -y
sudo apt install virtualenv -y

mkdir mlflow && cd mlflow
pipenv install mlflow awscli boto3
pipenv shell
```

3. **Configure AWS CLI**:

```bash
aws configure
```

4. **Start MLflow server**:

```bash
mlflow server \
  --host 0.0.0.0 \
  --default-artifact-root s3://<your-s3-bucket>
```

5. **Set MLflow tracking URI locally**:

```bash
export MLFLOW_TRACKING_URI=http://<EC2-PUBLIC-DNS>:5000
```

---

## DVC Pipelines

1. Initialize DVC:

```bash
dvc init
```

2. Define pipeline stages (`dvc.yaml`):

* **data_ingestion** → **data_preprocessing** → **model_building** → **model_evaluation** → **model_registration**

3. Run pipeline:

```bash
dvc repro
dvc dag
```

> After model evaluation, metrics and artifacts are saved in the S3 bucket.
> The model is registered in MLflow using the run ID.

---

## Backend & API

* Built with **FastAPI**.

* Loads model from **MLflow Model Registry**.

* Serves endpoints for:

  * Predictions with timestamps
  * Sentiment charts
  * Word clouds
  * Trend graphs

* Test with **Postman** or browser requests:

```bash
http://localhost:5000/predict_with_timestamps
```

---

## Chrome Extension

* Connects to YouTube API securely via your backend.

* Features:

  * Fetch top 500 comments for a video
  * Sentiment analysis
  * Comment summary metrics
  * Sentiment pie chart
  * Trend graph
  * Word cloud visualization
  * Top 25 comments with sentiment

* Load the extension in Chrome:
  `chrome://extensions → Load unpacked → select chrome_extension folder`.

---

## Docker Deployment

1. Build Docker image:

```bash
docker build -t youtube-sentiment:latest .
```

2. Run container:

```bash
docker run -p 8000:8000 youtube-sentiment:latest
```

---

## AWS CI/CD with GitHub Actions

1. **Create IAM user** with policies:

   * `AmazonEC2ContainerRegistryFullAccess`
   * `AmazonEC2FullAccess`

2. **Create ECR repository**:

```text
URI: <account_id>.dkr.ecr.<region>.amazonaws.com/mlproject
```

3. **Setup EC2**:

   * Install Docker:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
```

4. **Configure self-hosted GitHub runner** on EC2.

5. **Add GitHub secrets**:

| Secret Name           | Value           |
| --------------------- | --------------- |
| AWS_ACCESS_KEY_ID     | <from IAM>      |
| AWS_SECRET_ACCESS_KEY | <from IAM>      |
| AWS_REGION            | eu-north-1      |
| AWS_ECR_LOGIN_URI     | <ECR login URI> |
| ECR_REPOSITORY_NAME   | mlflow-ecr      |

6. **CI/CD Workflow** (`.github/workflows/cicd.yaml`):

   * Build Docker image
   * Push to ECR
   * Pull & run image on EC2

---

### Notes

* Make sure **AWS ports** (e.g., 5000 for MLflow, 8000 for API) are open in Security Groups.
* Always test pipelines locally with `dvc repro` before deploying.
* Keep your **API keys** secure in `.env` or backend service.




