SSH_URL="git@github.com:Ezibenroc/g5k_test.git"

echo "###  PROCESSING"
cd /tmp
mkdir -p notebooks && cd notebooks
cashew test --output . all && cd .. || exit 1
git clone $SSH_URL repository
cd repository
# Let's keep a single commit in this repository, we do not need history, so better to save some space
rm -rf .git
git init
mv ../*.html .
git add .
git config user.email "CI_test@$(hostname)"
git config user.name "CI-test"
git commit -am "[AUTOMATIC COMMIT] Regenerate notebooks"
git remote add origin $SSH_URL
git push --set-upstream origin master --force

echo "DONE: $0"
