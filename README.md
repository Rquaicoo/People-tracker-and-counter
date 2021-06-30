# People Tracker and Counter
The is a computer vision application that counts people entering and exiting an establishment. 

## Language and Libraries
- Python 3.8
- OpenCV

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
Command line arguments
```bash
-p, or --prototxt, => path to Caffe 'deploy' prototxt file. It is required.
-m, or --model, => path to Caffe 'pre-trained model'. It is required.
-i, or --input, => path to optional input video file. Leave it to use camera feed.
-o, or --output, => path to output video file. It is optional.
-c, or --confidence, => type =float, default = 0.4, probability threshold for detections.
-s, or --skip_frames, => type =int, default=30, number frames skippped between detections.
```

```bash
python counter.py --prototxt ./mobilenet_ssd/MobileNetSSD_deploy.prototxt \
	--model ./mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
	--input ./videos/example_01.mp4 --output output/output_01.avi
```

Using camera.
```bash
python counter.py --prototxt ./mobilenet_ssd/MobileNetSSD_deploy.prototxt \
	--model ./mobilenet_ssd/MobileNetSSD_deploy.caffemodel
```

### Results
The results from the execution is saved in ./results.csv in the format: (timestamp, totalUp, totalDown) whenever the count changes.