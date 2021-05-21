# NLP: IMapBook Collaborative Discussions Classification

## Quick start
Create a virtual environment and install dependencies from _requirements.txt_. You might also need to download some NLTK data, and potentially GloVe embeddings. Consult the _Additional dependencies_ section below if you encounter any errors.

To run the code first move to the _code_ folder.
```bash
cd code
```
Now you can perform TF-IDF / Logistic regression model evaluation by running the main script.
```bash
python main.py
```
To run any additional models just uncomment them the main file, but evaluation can take a long time (more than an hour for bert).

Notebook _custom_features.ipynb_ contains an analysis of custom features.

## Local environment setup
Create a virtual environment either with the help of your IDE or via shell.
```bash
virtualenv --python=python3.9 env
```
Activate it.
```bash
source env/bin/activate
```
Install the required dependencies.
```bash
pip install -r requirements.txt
```
To deactivate the environment run `deactivate`.

### Additional dependencies

You have to download NLTK data
```bash
python -c "import nltk; nltk.download('popular')"
```
and GloVe models.
```bash
mkdir -p models
wget -O models/glove.6B.zip http://nlp.stanford.edu/data/glove.6B.zip
```

## Installing Python dependencies
To install a new dependency, for example the _pandas_ library, add it to the _requirements.in_ file and install it either with `pip install pandas` or the following command.
```bash
pip install -r requirements.in
```
After installing, don't forget to regenerate the _requirements.txt_ file that contains all required dependencies and their versions.
```bash
pip freeze > requirements.txt
```