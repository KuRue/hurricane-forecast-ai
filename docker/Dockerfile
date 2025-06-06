# Hurricane Forecast AI - Development Docker Image
FROM nvidia/cuda:12.4.0-cudnn9-devel-ubuntu22.04

# Set environment variables
ENV PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    ca-certificates \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Create working directory
WORKDIR /app

# Install Conda (for complex geospatial dependencies)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create conda environment
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy \
    && rm /tmp/environment.yml

# Make the conda environment activation automatic
RUN echo "conda activate hurricane-forecast" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Copy project files
COPY requirements.txt /app/
COPY setup.py /app/
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Install the package in development mode
RUN conda activate hurricane-forecast && pip install -e .

# Create data directories
RUN mkdir -p /data/hurricane-forecast/{hurdat2,ibtracs,era5,models,checkpoints,cache,logs,mlruns}

# Set up Jupyter kernel
RUN conda activate hurricane-forecast && \
    python -m ipykernel install --user --name hurricane-forecast --display-name "Hurricane Forecast"

# Expose ports
EXPOSE 8000  # FastAPI
EXPOSE 8888  # Jupyter
EXPOSE 6006  # TensorBoard

# Set up entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
