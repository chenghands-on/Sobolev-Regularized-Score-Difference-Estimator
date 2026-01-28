# change it to your settings, if necessary.
export CUDA_HOME=/usr/local/cuda-12.3
export CPP_COMPILER=/bin/g++-11

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Uncomment the following line to enable debugging.
# export DEBUGGINGFLAG=-DDEBUG

nvcc -std=c++20 -shared $DEBUGGINGFLAG \
     -DLOGGING_OFF -O3 -o \
     PythonWrapper.so ml/PythonWrapperGradEst.cu \
     cpp/cumatrix.cu cpp/cukernels.cu \
     --extended-lambda -lopenblas -lcublas -lcurand \
     --compiler-options -fPIC -ccbin=$CPP_COMPILER

mv PythonWrapper.so ../core/