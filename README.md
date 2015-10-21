# detect_human
Detection of humans using HoG and SVM. 

## General Requirements
- Python 2.7.x
- numpy
- opencv 3
- scikit-learn

## Running
### detect_human.py
This script will train and test the class Human_Detector. 

- To train run: <br />
<code>python detect_human.py -t</code><br />
This will also save the Human_Detector class in a file called <code>human_detector.pkl</code>

- To test and load previously trained Human_Detector run:<br />
<code>python detect_human.py -l</code>

### detect_human2.py
This script will loop through images and detech humans using a sliding window approach

- To run: <br />
<code>python detect_human2.py</code>
