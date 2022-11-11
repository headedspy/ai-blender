import subprocess
import sys
import os
 
python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')
target = os.path.join(sys.prefix, 'lib', 'site-packages')
 
# upgrade pip
subprocess.call([python_exe, '-m', 'ensurepip'])
subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'])
 
# install tensorflow
subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'tensorflow', '-t', target])

# install cv2
subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'opencv-python', '-t', target])
 
print('TensorFlow installed')