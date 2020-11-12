#!/bin/bash

# run this on the deployment server to make it into a pushable git remote


main() {
    ask_if_sure "Did you set the paths in post-receive correctly?" || exit 1

    GIT_DIR=$(dirname "${0}")"/../"
    cd "${GIT_DIR}"

    # install hook
    echo "Installing post-receive hook"
    cp "deployment/post-receive" ".git/hooks/"
    chmod +x ".git/hooks/post-receive"

    # manage rights
    gitgroup="githost"
    echo "Giving all users in group ${gitgroup} write access"
    git config core.sharedRepository group
    chgrp -R "${gitgroup}" .
    sudo find . -type d -exec chmod g+s '{}' +
}

ask_if_sure() {
    read -p "${1} [yN]: " -n 1 -r
    echo ""
    if [[ ! ${REPLY} =~ ^[Yy]$ ]]; then
        echo "No"
        echo "Aborted"
        return 1
    else
        echo "Yes"
        echo "Proceeding..."
    fi
}

main $@
