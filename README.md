# People Tracker and Counter

The is a computer vision application that counts people entering and exiting an establishment.

## Language and Libraries

- Python 3.9
- OpenCV
- Flask

## Installation

Clone the repository and open it in an editor.

```bash
git clone https://github.com/Rquaicoo/People-tracker-and-counter.git
```

```bash
cd ./People-tracker-and-counter
pip install -r requirements.txt

```

## Running the Demo

### Running the Web Interface

The web interface was developed in Flask. To run it,

```bash
cd ./web
flask run
```

### Running the detector

Command line arguments

```bash

```

```bash
run the code and input the video path
```

Using camera.

```bash
run the code and press "w"
```

### Results

The results from the execution is saved in ./results.csv in the format: (timestamp, totalUp, totalDown) whenever the count changes.

