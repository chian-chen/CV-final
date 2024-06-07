# CV-final
---

## Real-Time Monitoring of Door Status in Public Transit Systems

> Group Number: 22
> 

> R12942181 陳祈安, B09901045 吳名洋, B09901082 楊秉嶧, B09901152 施文威
> 

---

## Project Overview:

This project aims to develop a generalized door detection system for public transit environments. The primary goal is to accurately monitor door statuses throughout various stages including closed, opening, open, and closing. The system is designed to be robust, effectively handling potential interferences such as:

- Occlusions by passengers
- Variations in lighting conditions (intense daylight to low nighttime light)
- Movement of the vehicle
- Reflections from glass surfaces

Our system has been evaluated across five testing videos featuring different perspectives and has consistently achieved 100% accuracy with our real-time algorithm.

### Performance Metrics for Testing Videos:

```bash
Time: 19.50867199897766 s (0.034225740349083614 s per frame)
Time: 4.710011959075928 s (0.014101832212802179 s per frame)
Time: 4.312044858932495 s (0.014920570446133201 s per frame)
Time: 6.917326211929321 s (0.015169575026160792 s per frame)
Time: 3.1110429763793945 s (0.01234540863642617 s per frame)
```

---

## Getting Started:

### **Dependencies:**

To ensure the proper functioning of the software, the following dependencies are required:

```bash
python>=3.10
numpy==1.26.4
opencv-python==4.10.0.82
matplotlib==3.9.0
```

A `requirements.txt` file is provided for easy installation. Follow these step-by-step instructions to set up the environment using conda:

```bash
git clone https://github.com/chian-chen/CV-final.git
cd CV-final
conda create --name CV-final-22 python=3.10
conda activate CV-final-22
pip install -r requirements.txt
```

### Running the Program:

Execute the program using the following command:

```bash
python3 algorithm.py --path $1 --output_filename $2
```

Where:

- `$1` is the path to the testing videos (default: `"./tests"`)
- `$2` is the name of the output JSON file (default: `"algorithm_output.json"`)