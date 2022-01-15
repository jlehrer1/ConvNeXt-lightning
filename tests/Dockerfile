FROM pytorch/pytorch

WORKDIR /src

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN apt-get --allow-releaseinfo-change update && \
    apt-get install -y --no-install-recommends \
        curl \
        sudo \
        vim 

RUN conda install --yes boto3 pandas numpy pip plotly scipy 
RUN pip install pytorch-lightning comet_ml scikit-learn Pillow torch torchvision torchmetrics

COPY . .