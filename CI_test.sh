TOKEN_PATH="GITLAB_TOKEN"
REPO_URL="https://gitlab.in2p3.fr/cornebize/g5k_test.git"
SSH_URL="git@gitlab.in2p3.fr:cornebize/g5k_test.git"

token=$(cat $TOKEN_PATH)
if [ $? -ne 0 ] ; then
    echo "Missing gitlab token, exiting"
    exit 1
fi

echo "###  PROCESSING"
cd /tmp
base_url=$(echo $REPO_URL | awk -F '//' '{ printf $2; }')
remote_url="https://oauth2:$token@$base_url"
test -d repository || GIT_LFS_SKIP_SMUDGE=1 git clone $SSH_URL repository --depth 1
cd repository
git config user.email "CI_test@$(hostname)"
git config user.name "gitlab-CI-test"
cd notebooks && cashew test --output . all && cd .. || exit 1
mkdir -p public && mv notebooks/*.html public || exit 1
git add notebooks public && git commit -m "[AUTOMATIC COMMIT] Generating test notebooks"
git push $SSH_URL
echo "DONE: $0"
