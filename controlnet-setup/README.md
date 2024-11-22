# Setup ControlNet:

1. Clone the repository:
```bash
pushd ..
git clone git@github.com:lllyasviel/ControlNet.git
popd
```

2. Copy `setup.py` and `requirements-controlnet.txt` to the root of the ControlNet repository:
```bash
cp controlnet-setup/setup.py ../ControlNet/
cp controlnet-setup/requirements-controlnet.txt ../ControlNet/
```

3. Make sure the venv is activated:
```bash
source venv/bin/activate
```

4. pip install ControlNet:
```bash
pip install -e ../ControlNet
```