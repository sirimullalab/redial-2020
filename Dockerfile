FROM informaticsmatters/rdkit-python3-debian:Release_2020_03_1
USER ${UID}:${GID}

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --allow-releaseinfo-change && apt-get install -y libpango1.0-0 \
    libcairo2 \
    libpq-dev \
    perl \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# MAINTAINER
MAINTAINER Srijan Verma<vermasrijan44@gmail.com>
COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY saved_models ./saved_models
COPY static ./static
COPY scalers ./scalers
COPY smi_all_dict_updated_new_cleaning_with_3cl.pkl drug_central_drugs-stand.csv run_script.py config.py app.py ./
COPY mayachemtools ./mayachemtools
# For webapp
ENTRYPOINT ["python3", "app.py"]
EXPOSE 5000
