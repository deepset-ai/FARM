FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

RUN apt-get update && apt-get install -y git

# Setup locales
RUN apt-get update \
    	&& apt-get install -y --no-install-recommends \
    		locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR /home/user

# Install apex
RUN git clone https://github.com/NVIDIA/apex \
    && cd apex \
    && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./



# Install FARM
COPY setup.py requirements.txt readme.rst /home/user/
RUN pip install -r requirements.txt
COPY farm farm
RUN pip install -e .


# Copy Training Scripts
COPY examples examples

CMD FLASK_APP=farm.inference_rest_api flask run --host 0.0.0.0
