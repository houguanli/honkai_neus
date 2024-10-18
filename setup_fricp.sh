pip install pybind11-stubgen
cmake -B build .
cd build
make -j4
cd ..
cp build/py_fricp.* ./models
pybind11-stubgen build.py_fricp
mv stubs/build/py_fricp.* ./models
rm -rf stubs