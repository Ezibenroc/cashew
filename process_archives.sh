#! /usr/bin/env bash

git config user.email "CI@$(hostname)"
git config user.name "gitlab-CI"

git checkout master
git pull

for branch in $(git branch -r | grep exp_) ; do
    commit=$(git show --format="%H" $branch | head -n 1)
    echo "Processing branch $branch (commit $commit)"
    git cherry-pick $commit
done

mkdir -p data

for i in new_data/* ; do
    echo Processing file $i
    cashew $i result.csv data.db dgemm
    mv $i data
done
rmdir new_data

git add data data.db
git rm -r new_data
git commit -m "[AUTOMATIC COMMIT] Processing archive(s)"
