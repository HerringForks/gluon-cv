pip install pybind11==2.5.0

# Install NVIDIA COCO and dllogger
pip uninstall -y pycocotools
pip --no-cache-dir --no-cache install \
    'git+https://github.com/NVIDIA/cocoapi#egg=pycocotools&subdirectory=PythonAPI'

# Install nightly GluonCV for now since changes in Mask RCNN is not compatible with stable pip release
pip uninstall -y gluoncv
pip install gluoncv==0.8.0b20200801
