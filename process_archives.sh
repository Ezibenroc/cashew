#! /usr/bin/env bash

git config user.email "CI@$(hostname)"
git config user.name "gitlab-CI"

git checkout master
git fetch
git reset --hard origin/master

rm -f processed_branches
for branch in $(git branch -r | grep exp_) ; do
    commit=$(git show --format="%H" $branch | head -n 1)
    echo "Processing branch $branch (commit $commit)"
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
    echo "Processing file $f"
    cashew $f result.csv data.db --compression zlib --compression_lvl 9 --format table
    mv $f data
done
rmdir new_data

git add data data.db
git rm -r new_data
git commit -m "[AUTOMATIC COMMIT] Processing archive(s)"
