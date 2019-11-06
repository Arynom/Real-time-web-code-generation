# Setup

## 1. Setup Python

Install **python 3** on your computer.<br>
You can download Python 3.7 from [here](https://www.python.org/downloads/release/python-374/) and install it on your machine.

## 2. Install required packages

If you want to run this python script, some packages need to be installed before. **numpy** is required to manipulate image data. To recognize images using the trained model, **tensorflow** package should be installed on your computer.

**opencv-python** should be installed to use your webcam.

You can easily install all the dependant packages using the following command.

`pip install -r requirements.txt`

And you need to install **object_detection** on the local project.

`python setup.py install`

## 3. Setup React.js

In order to look at the dynamically generated part of the code, you need to install **React.js** since the demo application was made using [React.js](https://reactjs.org/).

First of all, you need to install **Node.js** on your computer. You can download Node.js [here](https://nodejs.org/en/download/) and install it.

Open the command prompt and execute the following commands to install **React.js** on your computer.

`npm install -g create-react-app`

## 4. Install Node dependencies

To generate a live demo page, a React.js project was made and it needs to be installed required dependencies on your computer in order to run the demo application.

Go into the **preview** directory in the project folder and run the following commands.

`npm install`

# Test

You can execute the python script using following command in command prompt or terminal.

`python live-demo.py`

If you want to see the automatically generated React component, just **double click** the `run-demo.bat` to execute the React project.
