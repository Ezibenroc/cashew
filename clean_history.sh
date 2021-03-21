set -u
set -e

echoerr() { echo "$@" 1>&2; }

old_remote=$1
new_remote=$2

cd /tmp
git clone $old_remote repo --depth 1
cd repo

rm -rf .git

git init
git config user.email "clean_history@$(hostname)"
git config user.name "clean_history"
git remote add origin $new_remote

git add README*
git commit -m "First commit"
git push --set-upstream origin master

wget https://raw.githubusercontent.com/Ezibenroc/cashew/master/split_hdf5.py -O ~/split_hdf5.py
python3 ~/split_hdf5.py data.db 2019 1
mv data.db /tmp/old_data.db
mv $(ls data_20*db | tail -n 1) data.db
python3 ~/split_hdf5.py data_monitoring.db 2019 1
mv data_monitoring.db /tmp/old_data_monitoring.db
mv $(ls data_monitoring_20*db | tail -n 1) data_monitoring.db

for f in data*.db; do
    git add $f
    git commit -m "File $f"
    git push || git push || git push || git push
done

for year in {2019,2020,2021}; do
    for month in {01..12} ; do
        git add data/*${year}-${month}*.zip
        git commit -m "Archives of ${year}-${month}"
        git push || git push || git push || git push
    done
done

git add .
git commit -am "Last commit"
git push || git push || git push || git push
