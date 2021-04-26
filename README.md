# NLP: IMapBook Collaborative Discussions Classification

# Local environment setup
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

You have to download _nltk_ data
```bash
python -c "import nltk; nltk.download('popular')"
```
and _glove_ models.
```bash
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