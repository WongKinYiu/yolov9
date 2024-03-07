# YOLOv9u

# Start FROM PyTorch image https://hub.docker.com/r/pytorch/pytorch or nvcr.io/nvidia/pytorch:23.03-py3
FROM nvcr.io/nvidia/pytorch:23.03-py3
RUN pip install --no-cache nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install linux packages
# g++ required to build 'tflite_support' and 'lap' packages, libusb-1.0-0 required for 'tflite_support' package
RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0

# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade --no-install-recommends -y openssl tar

# Create working directory
WORKDIR /usr/src/yolov9u

# Copy contents
# COPY . /usr/src/yolov9u  # git permission issues inside container
RUN git clone https://github.com/WongKinYiu/yolov9u -b yolov9u /usr/src/yolov9u
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt /usr/src/yolov9u/

# Install pip packages
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache -e ".[export]" albumentations comet pycocotools

# Fix error: `np.bool` was a deprecated alias for the builtin `bool` segmentation error in Tests
RUN pip install --no-cache numpy==1.23.5
# Remove exported models
RUN rm -rf tmp

# Set environment variables
ENV OMP_NUM_THREADS=1
# Avoid DDP error "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library" https://github.com/pytorch/pytorch/issues/37377
ENV MKL_THREADING_LAYER=GNU
