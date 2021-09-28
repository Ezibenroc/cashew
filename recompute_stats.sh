set -u
set -e

echoerr() { echo "$@" 1>&2; }

remote=$1

cd /tmp
git clone $remote repo --depth 1
cd repo

git config user.email "reprocess@$(hostname)"
git config user.name "reprocess"

git mv stats.csv stats_legacy.csv
git mv stats_monitoring.csv stats_monitoring_legacy.csv

git commit -m """[AUTOMATIC COMMIT] Archiving the stats files"""
git push || git push || git push || git push

for f in data_20*.db data.db; do
    cashew stats $f stats.csv
    git add .
    git commit -m """[AUTOMATIC COMMIT] Recomputing the statistics for data ${f}"""
    git push || git push || git push || git push
done
for f in data_monitoring_20*.db data_monitoring.db; do
    cashew stats $f stats_monitoring.csv
    git add .
    git commit -m """[AUTOMATIC COMMIT] Recomputing the statistics for data ${f}"""
    git push || git push || git push || git push
done
