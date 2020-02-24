#! /usr/bin/env bash

echoerr() { echo "$@" 1>&2; }

git config user.email "CI@$(hostname)"
git config user.name "gitlab-CI"

git checkout master
git fetch --prune
git reset --hard origin/master

rm -f processed_branches
for branch in $(git branch -r | grep exp_) ; do
    commit=$(git show --format="%H" $branch | head -n 1)
    echoerr "Processing branch $branch (commit $commit)"
    echo $branch | cut -d/ -f2 >> processed_branches
    git cherry-pick $commit
done

nb_archives=$(mkdir -p new_data && ls new_data | wc -l)

if [ $nb_archives -eq 0 ] ; then
    echo "No new archive, aborting."
    rmdir new_data
    exit 0
fi

mkdir -p data

for f in new_data/* ; do
    echoerr "Processing file $f"
    cashew extract $f performance data.db --compression zlib --compression_lvl 9 --format table
    cashew extract $f monitoring data_monitoring.hdb --compression zlib --compression_lvl 9 --format table
    mv $f data
done
rmdir new_data

cashew stats data.db stats.csv
cashew stats data_monitoring.hdb stats_monitoring.csv

git add data data.db stats.csv
git rm -r new_data
git commit -m "[AUTOMATIC COMMIT] Processing archive(s)"
