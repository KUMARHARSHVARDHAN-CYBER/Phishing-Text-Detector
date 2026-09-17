# Probabilistic Phishing & Malicious Text Detector

Overview
This project is an automated cyber security tool using Machine Learning and Probability Theory to classify text messages or emails as malicious or safe. It implements the principles of Bayes' theorem to classify text messages and emails in the realm of natural language processing.

Features
Ingestion and cleaning of data from CSV files
Transforming text into TF-IDF matrices of terms
Classifying text using Multinomial Naive Bayes
Generating a confusion matrix that calculates the sensitivity and specificity of the model
Technology

Python 3. x

Libraries: Pandas, Scikit-learn, NLTK
How to use
Clone this repo to your local machine
Install the dependencies by running pip install pandas scikit-learn nltk
Get the dataset (e.g. sms_spam.csv) and save it in the root folder
Lastly, run python app.py

Testing
The script already splits the data into an 80/20 training/test set, so you can use that to test the accuracy of the model on data it has not seen before
You can also test it yourself by modifying the test_messages list at the bottom of app.py with your own strings and see what the probability is that it will be classified as spam or not spam.
