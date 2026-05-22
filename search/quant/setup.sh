cd ft
python setup_cuda.py develop
cd ..

cd AutoGPTQ
pip install -e .
cd ..

cd hqq
pip install -e .
