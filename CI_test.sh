TOKEN_PATH="GITLAB_TOKEN"
REPO_URL="https://gitlab.in2p3.fr/tom.cornebize/g5k_data.git"

token=$(cat $TOKEN_PATH)
if [ $? -ne 0 ] ; then
    echo "Missing gitlab token, exiting"
    exit 1
fi

echo "###  PROCESSING"
cd /tmp
base_url=$(echo $REPO_URL | awk -F '//' '{ printf $2; }')
remote_url="https://oauth2:$token@$base_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $REPO_URL repository --depth 1
git config user.email "CI_test@$(hostname)"
git config user.name "gitlab-CI-test"
cd repository
cashew test --output notebooks && git add notebooks && git commit -m "[AUTOMATIC COMMIT] Generating test notebooks"
git push $remote_url
