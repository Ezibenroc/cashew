sudo-g5k apt update
sudo-g5k apt remove -y python3-pandas # the debian package is very outdated...
sudo-g5k apt install --no-install-recommends -y git-lfs libhdf5-dev libhdf5-serial-dev
sudo-g5k pip3 install https://github.com/Ezibenroc/cashew/releases/download/0.2.4/cashew-0.2.4-py3-none-any.whl
cashew --git-version
sudo-g5k wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/process_archives.sh -O /bin/process_archive
sudo-g5k chmod +x /bin/process_archive
sudo-g5k wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/CI.sh -O /CI.sh
sudo-g5k wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/CI_test.sh -O /CI_test.sh
sudo-g5k wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/reprocess_all.sh -O /reprocess_all.sh
echo "DONE: $0"
