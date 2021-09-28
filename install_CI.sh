sudo-g5k apt update
sudo-g5k apt remove -y python3-pandas # the debian package is very outdated...
sudo-g5k apt install --no-install-recommends -y git-lfs libhdf5-dev libhdf5-serial-dev
sudo-g5k pip3 install 'pyparsing>=2.2.1'  # otherwise, fail with "ImportError: Matplotlib requires pyparsing>=2.2.1; you have 2.2.0"
sudo-g5k pip3 install https://github.com/Ezibenroc/cashew/releases/download/0.2.11/cashew-0.2.11-py3-none-any.whl
cashew --git-version
sudo-g5k wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/process_archives.sh -O /bin/process_archive
sudo-g5k chmod +x /bin/process_archive
sudo-g5k wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/CI.sh -O /CI.sh
sudo-g5k wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/CI_test.sh -O /CI_test.sh
sudo-g5k wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/reprocess_all.sh -O /reprocess_all.sh
sudo-g5k wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/clean_history.sh -O /clean_history.sh
sudo-g5k wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/upload_zenodo.sh -O /upload_zenodo.sh
sudo-g5k wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/recompute_stats.sh -O /recompute_stats.sh
echo "DONE: $0"
