SSH_URL="git@github.com:Ezibenroc/g5k_test.git"

echo "###  PROCESSING"
cd /tmp
rm -rf notebooks
cashew test --output /tmp/notebooks all || exit 1
git clone $SSH_URL notebook_repository
cd notebook_repository
# Let's keep a single commit in this repository, we do not need history, so better to save some space
rm -rf .git
git init
mv ../notebooks/*.html .
git add .
git config user.email "CI_test@$(hostname)"
git config user.name "CI-test"
git commit -am "[AUTOMATIC COMMIT] Regenerate notebooks"
git remote add origin $SSH_URL
git push --set-upstream origin master --force

echo "DONE: $0"
