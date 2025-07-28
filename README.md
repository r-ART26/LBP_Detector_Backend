# Flask LBP Face Blur App

This project is a web-based application that performs real-time face detection and blurring using a custom-trained Local Binary Pattern (LBP) cascade classifier. The system is implemented with Python and Flask, and processes live video streamed from an IP camera (such as an Android device using IP Webcam).

It provides visual feedback by displaying:
- Blurred detected faces on the video stream
- Frames per second (FPS)
- Memory usage
- Bandwidth usage during streaming

> ⚠️ **Important**: The application requires OpenCV version **3.4.20**, as this is the last version that fully supports the `opencv_traincascade` and `opencv_createsamples` tools needed for training LBP or Haar cascade classifiers. Later versions of OpenCV (4.x and above) have deprecated or removed key functionalities.

## 🔧 Installation & Requirements

Before running the application, make sure your environment meets the following requirements:

### 🛠️ Requirements

- Python 3.8 or later
- OpenCV **3.4.20** (critical for compatibility with LBP cascade training tools)
- Flask
- psutil
- A video source (e.g., IP Webcam app or other RTSP/HTTP stream)

We recommend using a virtual environment to isolate dependencies.

### 📦 Install Dependencies

```bash
# Clone this repository
git clone https://github.com/r-ART26/LBP_Detector_Backend.git
cd LBP_Detector_Backend

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # For Linux/macOS
# .\venv\Scripts\activate  # For Windows

# Install required Python packages
pip install -r requirements.txt
```
## 🧠 Training the LBP Classifier

The face detection classifier used in this system was trained using the `opencv_traincascade` tool from OpenCV **3.4.20**, with the following key parameters:

- **Number of positive samples (`-numPos`)**: 4000  
- **Number of negative samples (`-numNeg`)**: 3900  
- **Number of training stages (`-numStages`)**: 15  
- **Window size (`-w` and `-h`)**: 24 x 24 pixels  
- **Feature type (`-featureType`)**: LBP (Local Binary Patterns)  
- **Precalculated value buffer size (`-precalcValBufSize`)**: 3048 MB  
- **Precalculated index buffer size (`-precalcIdxBufSize`)**: 3048 MB  
- **Minimum hit rate (`minHitRate`)**: 0.995  
- **Maximum false alarm rate (`maxFalseAlarmRate`)**: 0.5  
- **Boosting type (`boostType`)**: GAB (Gentle AdaBoost)  
- **Maximum tree depth (`maxDepth`)**: 1  
- **Maximum number of weak classifiers per stage (`maxWeakCount`)**: 100  


