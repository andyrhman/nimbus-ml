# Nimbus Machine Learning Server

<p align="center">
  <img src="https://1000logos.net/wp-content/uploads/2020/08/Django-Logo.png" width="300" height="200" />
</p>

## Introduction

This is the backend server responsible for managing the machine learning model. It runs on TensorFlow 2.18 for processing the model. This backend server communicates with our NodeJS backend server, where NodeJS handles the request to this Django REST server for processing the model and subsequently sends the result back to NodeJS.

## Features

Here are the features of our backend server Machine Learning API:
- Travel Destination Recommendation by Distance
- Travel Destination Recommendation by Rating
- Itinerary Recommendation using Machine Learning

## First Time Set Up & Configuration

Install the Django Web Framework:

```bash
sudo apt update && sudo apt install python3.10 python3.10-venv python3.10-dev
sudo apt update && sudo apt install libgl1-mesa-glx
pip install django
pip install djangorestframework   
pip install django-filter
pip install python-decouple
pip install pandas
pip install numpy
pip install tensorflow
```

Create the directory:

```bash
mkdir nimbus-ml
django-admin startproject app .
django-admin startapp <folder name>

```