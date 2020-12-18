set -u
set -e

echoerr() { echo "$@" 1>&2; }

old_remote=$1
new_remote=$2

cd /tmp
git clone $old_remote repo --depth 1
cd repo

rm -rf .git data*db stats*csv

for f in data/* ; do
    echoerr "Processing file $f"
    cashew extract $f performance data.db --compression zlib --compression_lvl 9 --format table
    cashew extract $f monitoring data_monitoring.db --compression zlib --compression_lvl 9 --format table || echoerr "    Failed to extract monitoring data"
done

cashew stats data.db stats.csv
cashew stats data_monitoring.db stats_monitoring.csv

git init
git config user.email "reprocess@$(hostname)"
git config user.name "reprocess"
git add .
git commit -m """[AUTOMATIC COMMIT] Reprocessing all the archives"""
git remote add origin $new_remote
git push --set-upstream origin master
