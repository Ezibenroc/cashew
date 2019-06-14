#! /usr/bin/env bash

git checkout master
for i in $(git branch -r | grep exp_) ; do
    echo Rebasing branch $i
    git rebase $i
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
git commit --author "gitlab-CI <CI@$(hostname)>" -m "[AUTOMATIC COMMIT] Processing archive(s)"
