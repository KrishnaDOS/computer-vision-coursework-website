# Computer Vision Homework Site

This repo contains the code and materials for the Computer Vision Homework Site.  

## Setup venv
This website will not work without the required python packages. Python version must be 3.12 - don't go any higher.  

Setup a virtual environment like so:  

```bash
# From project root
./setup_venv/setup_venv.sh #for unix systems
```

```ps1
# From project root
.\setup_venv\setup_venv.ps1 #for windows
```

## Launch site in debug
After setting up the venv, you can launch the site in debug mode like so:

```bash
# From project root
python3 app.py
```

## Launch site in production
Setup gunicorn (unix) or waitress (windows).

### Running with Gunicorn

You can use the prepared `gunicorn_config.py` file in the project root. Example usage:

```bash
# From the project root
gunicorn -c gunicorn_config.py app:app       # run with config file
```

To run in the background use:

```bash
gunicorn -c gunicorn_config.py app:app --daemon
```

### Running with Waitress
Instructions on how to use waitress to run a Flask app can be found on the internet.
