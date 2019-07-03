#! /usr/bin/env bash

git config user.email "CI@$(hostname)"
git config user.name "gitlab-CI"

git checkout master
git fetch
git reset --hard origin/master

no_archive=true
for branch in $(git branch -r | grep exp_) ; do
    commit=$(git show --format="%H" $branch | head -n 1)
    echo "Processing branch $branch (commit $commit)"
    git cherry-pick $commit
    no_archive=false
done

if $no_archive ; then
    echo "No new archive, aborting."
    exit 0
fi

mkdir -p data

for f in new_data/* ; do
    echo "Processing file $f"
    cashew $f result.csv data.db dgemm
    mv $f data
done
rmdir new_data

git add data data.db
git rm -r new_data
git commit -m "[AUTOMATIC COMMIT] Processing archive(s)"
