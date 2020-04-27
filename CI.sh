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
git clone $REPO_URL repository --depth 1
cd repository
git remote set-branches origin '*'  # https://stackoverflow.com/a/27393574
git fetch -v
process_archive || exit 1
git lfs push --all $remote_url && git push $remote_url || exit 1
for branch in $(cat processed_branches) ; do git push $remote_url --delete $branch ; done
