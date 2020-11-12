#!/bin/bash

# run this on the deployment server to make it into a pushable git remote

GIT_GROUP="githost"

main() {
    CHECKED_OUT_DIR=$(dirname "${0}")"/../"
    # source the GIT_DIR variable from the hook
    source "${CHECKED_OUT_DIR}/deployment/post-receive"
    if [ -z "${GIT_DIR}" ]; then
        echo "GIT_DIR not set!"
        exit 1
    fi
    ask_if_sure "Did you set the paths in post-receive correctly?" || exit 1

    # install hook
    echo "Installing post-receive hook"
    cp "${CHECKED_OUT_DIR}/deployment/post-receive" "${GIT_DIR}/hooks/"
    chmod +x "${GIT_DIR}/hooks/post-receive"

    # manage rights
    echo "Giving all users in group ${GIT_GROUP} write access"
    chgrp -R "${GIT_GROUP}" "${GIT_DIR}"
    sudo find "${GIT_DIR}" -type d -exec chmod g+s '{}' +
    git --git-dir="${GIT_DIR}" config core.sharedRepository group
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
