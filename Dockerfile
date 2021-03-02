# Docker image generator for manylinux. See:
# https://github.com/pypa/manylinux/tree/manylinux1

# Build with: (note the -t hirola-manylinux1_x86_64 is just an arbitrary name)
#  $ docker build -t hirola-manylinux1_x86_64 .
# Or to specify an alternative base (say manylinux1_i686 for 32bit Linux):
#  $ docker build -t hirola-manylinux1_i686 --build-arg BASE=manylinux1_i686 .
# Then boot into your new image with:
#  $ docker run -it hirola-manylinux1_x86_64
# The above launches bash inside the image. You can append arbitrary shell
# commands to run those instead such as the following to launch Python:
#  $ docker run -it hirola-manylinux1_x86_64 python
# Or to run pytest:
#  $ docker run -it hirola-manylinux1_x86_64 pytest
# This repo is copied into the image on `docker build` into the path /io which
# is also the default cwd. To instead mount this repo so that file changes are
# syncronised use:
#  $ docker run -v `pwd`:/io -it hirola-manylinux1_x86_64

ARG BASE=manylinux1_x86_64
FROM quay.io/pypa/${BASE}
# Choosing a Python version is done just by prepending its bin dir to PATH.
ENV PATH=/opt/python/cp39-cp39/bin:$PATH
# Install dependencies. Do this before COPY to encourage caching.
RUN pip install --prefer-binary wheel auditwheel sloth-speedtest numpy
# Copy across this repo.
COPY . /io
# Set the repo's root as the cwd.
WORKDIR /io
# Install it.
RUN pip install --prefer-binary -e .[test]
