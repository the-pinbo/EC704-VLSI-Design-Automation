# How to run Jupyter Notebook on Windows and Linux

## Windows

### Install Python:

Jupyter notebook runs on Python, so you'll need to have Python installed on your system. You can download and install the latest version of Python from the official Python website: https://www.python.org/downloads/

### Install Jupyter Notebook:

Once you have Python installed, you can install Jupyter notebook using the following command in the Command Prompt:

`bash
pip install jupyter
`

### Create a Virtual Environment

To keep your dependencies organized, it's recommended to create a virtual environment. You can create a virtual environment using the following command:

`bash
python -m venv .venv
`

### Activate the Virtual Environment:

To activate the virtual environment, run the following command in the Command Prompt:

`bash
.venv\Scripts\activate
`

### Install Dependencies:

You can install all the dependencies using the requirements.txt file using the following command:

`pip install -r requirements.txt`

### Launch Jupyter Notebook:

Finally, you can launch Jupyter Notebook using the following command:

`jupyter notebook`

## Linux

### Install Python:

Jupyter notebook runs on Python, so you'll need to have Python installed on your system. You can install Python using the following command in the terminal:

`sudo apt-get install python3`

### Install Jupyter Notebook:

Once you have Python installed, you can install Jupyter notebook using the following command in the terminal:

`pip3 install jupyter`

### Create a Virtual Environment:

To keep your dependencies organized, it's recommended to create a virtual environment. You can create a virtual environment using the following command:

`python3 -m venv .venv`

### Activate the Virtual Environment:

To activate the virtual environment, run the following command in the terminal:

`source .venv/bin/activate`

### Install Dependencies:

You can install all the dependencies using the requirements.txt file using the following command:

`pip install -r requirements.txt`

### Launch Jupyter Notebook: Finally, you can launch Jupyter Notebook using the following command:

`jupyter notebook`
