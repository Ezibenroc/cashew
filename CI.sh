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

# There is an issue somewhere, maybe with git LFS, maybe with IN2P3's Gitlab, I don't know.
# If we do only 'git push', then the command fails and tell us to do 'git lfs push --all'.
# If we do 'git lfs push --all', then the command does not terminate (it seems).
# But, if start by 'git lfs push --all' in the background and wait a bit, then 'git push' works fine.
# So, this is what the following does. It is ugly, but I am simply too lazy to fill in a bug report.
git lfs push --all $remote_url &
sleep 60 && git push $remote_url || exit 1
for branch in $(cat processed_branches) ; do git push $remote_url --delete $branch ; done
