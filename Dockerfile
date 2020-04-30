FROM python:3.7-buster

RUN apt update && apt install --no-install-recommends -y \
    git-lfs \
    libhdf5-dev \
    libhdf5-serial-dev
RUN pip3 install https://github.com/Ezibenroc/cashew/releases/download/0.0.4/cashew-0.0.4-py3-none-any.whl
RUN cashew --git-version
RUN wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/process_archives.sh -O /bin/process_archive
RUN chmod +x /bin/process_archive
RUN wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/CI.sh -O /CI.sh
RUN wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/CI_test.sh -O /CI_test.sh

COPY GITLAB_TOKEN /
