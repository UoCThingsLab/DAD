rm -rf dataset
mkdir dataset
cd dataset
kaggle datasets download -d pesehr/driving-anomaly-detection
cd dataset
mkdir v0.1
unzip driving-anomaly-detection.zip -d ./v0.1
rm driving-anomaly-detection.zip

