pip install pybind11-stubgen
cmake -B build . -DCMAKE_BUILD_TYPE=Release
cd build
make -j4
cd ..
pybind11-stubgen build.py_fricp
cp build/py_fricp.* ./models
mv stubs/build/py_fricp.* ./models
rm -rf stubs
echo "Fricp setup complete"
echo "Run 'python example/py_fricp.py' to test the model"