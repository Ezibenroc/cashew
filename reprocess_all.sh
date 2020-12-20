set -u
set -e

echoerr() { echo "$@" 1>&2; }

old_remote=$1
new_remote=$2

cd /tmp
git clone $old_remote repo --depth 1
cd repo

rm -rf .git data*db stats*csv

git init
git config user.email "reprocess@$(hostname)"
git config user.name "reprocess"
git remote add origin $new_remote

for year in {2019,2020}; do
    for month in {01..12} ; do
        archives=false
        for f in $(find data -name "*${year}-${month}-*" | sort) ; do
            echoerr "Processing file $f"
            cashew extract $f performance data.db --compression zlib --compression_lvl 9 --format table
            cashew extract $f monitoring data_monitoring.db --compression zlib --compression_lvl 9 --format table || echoerr "    Failed to extract monitoring data"
            archives=true
        done
        if [ $archives = true ] ; then
            cashew stats data.db stats.csv
            cashew stats data_monitoring.db stats_monitoring.csv
            git add .
            git commit -m """[AUTOMATIC COMMIT] Reprocessing all the archives for ${year}-${month}"""
            git push --set-upstream origin master || git push || git push || git push
        fi
    done
done
