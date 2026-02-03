cd ..

uv python install 3.10

uv python pin 3.10

uv venv

uv lock
uv sync

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
git submodule update --init --recursive

uv pip install ninja
uv pip install wheel
set DISTUTILS_USE_SDK=1
uv pip install --no-build-isolation submodules/detectron2

pause
