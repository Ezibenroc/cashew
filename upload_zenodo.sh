# This script clones the two g5k_test repositories (the large one with all the data, the smaller one with the website).
# Then, it deletes the .git directories as we do not need the history.
# Finally, it makes a zip archives with both directories and upload it to zenodo.

set -u
set -e

echoerr() { echo "$@" 1>&2; }

data_repo=$1
website_repo=$2
zenodo_id=$3
zenodo_tokenpath=$4

cd /tmp
git clone $data_repo g5k_test
git clone $website_repo website
rm -rf {g5k_test,website}/.git
mv website g5k_test
zip -r g5k_test.zip g5k_test

git clone git@github.com:jhpoelen/zenodo-upload.git
token=$(cat $zenodo_tokenpath)
export ZENODO_TOKEN=$token
bash zenodo-upload/zenodo_upload.sh $zenodo_id g5k_test.zip
