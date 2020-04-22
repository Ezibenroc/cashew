curl https://raw.githubusercontent.com/Ezibenroc/cashew/master/Dockerfile | docker build --tag my_image -f - ./CI_G5K
docker run --name CI --tmpfs /tmp --rm -t my_image bash CI.sh
