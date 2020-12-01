#!/bin/bash

# run this on the deployment server to make it into a pushable git remote

: "${GIT_GROUP:=githost}"

main() {
    GIT_DIR=$(realpath "${1}")
    TARGET=$(dirname $(dirname $(dirname $(realpath "${0}"))))

    echo "GIT_DIR is \"${GIT_DIR}\""
    echo "TARGET is \"${TARGET}\""

    ask_if_sure "Are these correct?" || exit 1

    sed -e 's|TARGET=""|TARGET="'"${TARGET}"'"|' \
        -e 's|GIT_DIR=""|GIT_DIR="'"${GIT_DIR}"'"|' \
        "${TARGET}/deployment/jupyterlab/post-receive" > "/tmp/post-receive"

    # install hook
    echo "Installing post-receive hook"
    cp "/tmp/post-receive" "${GIT_DIR}/hooks/"
    chmod +x "${GIT_DIR}/hooks/post-receive"

    # manage rights
    echo "Giving all users in group \"${GIT_GROUP}\" write access to the repo (need sudo for that)"
    sudo chgrp -R "${GIT_GROUP}" "${GIT_DIR}"
    sudo find "${GIT_DIR}" -type d -exec chmod g+s '{}' +
    git --git-dir="${GIT_DIR}" config core.sharedRepository group
}

ask_if_sure() {
    read -p "${1} [yN]: " -n 1 -r
    echo ""
    if [[ ! ${REPLY} =~ ^[Yy]$ ]]; then
        echo "No, aborted"
        return 1
    else
        echo "Yes, proceeding..."
    fi
}

main $@
