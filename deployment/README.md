 
## Jupyterlab single server

### Pushing to the jupyterlab server

Current parameters:

* `server`: `192.168.2.63`
* `repo`: `/home/agfalta/git_host/agfalta_tools.git`
* `deploy_dir`: `/home/agfalta/jupyterlab_notebooks/agfalta_tools` (see [below](Set-up-the-git-repos-on-the-server))
* unix group for users that can push to deployment: `githost`

If you want to push to the jupyterlab production server in the IFP group at Uni Bremen, you need to add this as a separate remote (call it `deployment`). You also need ssh access to the machine and a user that is in the `githost` group so you have write access to the server. 

```sh
$ git remote add deployment user@server:repo/agfalta_tools.git
```

Now, when you do `git pull` you will get your files from github. For pushing, you can do either `git push --tags` to push to github or `git push deployment` to push to the IFP server.

##### Fancy way:

If you want to do it fancy, you can push to both remotes simultaneously with [this](https://stackoverflow.com/questions/5785549/able-to-push-to-all-git-remotes-with-the-one-command):

```sh
$ git remote add all git@github.com:surf-sci-bc/agfalta_tools.git
$ git remote set-url --push --add all user@server:repo/agfalta_tools.git
$ git remote set-url --push --add all git@github.com:surf-sci-bc/agfalta_tools.git
```

And then push both to origin and deployment with

```sh
$ git push all
```


### Set up the git repos on the server

All commands are on the server, so ssh there first. Don't forget to adjust the paths `repo` and `deploy_dir`. The last command is important as it installs the post-receive hook and sets the file permissions correctly (for that it asks for the `sudo` password).

```sh
$ mkdir -p repo
$ git clone --bare https://github.com/surf-sci-bc/agfalta_tools.git repo/agfalta_tools.git
$ mkdir -p deploy_dir
$ git clone repo/agfalta_tools.git deploy_dir/agfalta_tools
$ sudo groupadd githost
$ deploy_dir/agfalta_tools/deployment/jupyterlab/deploy.sh repo/agfalta_tools.git
```

Now everything should be set and you can add `user@server:repo/agfalta_tools.git` as your deployment remote (see [above](#Pushing-to-the-jupyterlab-server)). For that, the `user` needs to be in the `githost` group (you can change this group in `deployment/deploy.sh`). 

You can now install `agfalta_tools` by executing these lines from the venv that you intend to use:

```sh
(venv) $ cd deploy_dir/agfalta_tools
(venv) $ python3 -m pip install -e .
```

##### Old method

*NOTE: Dont do this anymore*

If you need to set up the server again, you need to keep an up-to-date working copy on some machine. Then, create an empty repo in `/home/agfalta/git_host` (if you change this location, do update this manual). See also [this github gist](https://gist.github.com/noelboss/3fe13927025b89757f8fb12e9066f2fa).

```sh
$ ssh agfalta@192.168.2.63
(on deployment server) $ cd /home/agfalta/
(on deployment server) $ mkdir git_host && cd git_host
(on deployment server) $ git init --bare agfalta_tools.git
```

Then, copy the `post-receive` script from your local working copy `agfalta_tools/deployment/` into `agfalta_tools.git/hooks` and remember to make it executable:

```sh
(on deployment server) $ chmod +x post-receive
```

Make the repositories shared to avoid file permission problems (See also [this stackoverflow question](https://stackoverflow.com/questions/6448242/git-push-error-insufficient-permission-for-adding-an-object-to-repository-datab)):

```sh
(on deployment server) $ cd git_host/agfalta_tools.git/
(on deployment server) $ git config core.sharedRepository group
(on deployment server) $ sudo chown -R agfalta:githost ~/git_host
(on deployment server) $ sudo find ~/git_host -type d -exec chmod g+s '{}' +
```

From your local machine, add the bare server repo as deployment remote:

```sh
$ git remote add deployment username@192.168.2.63:/home/agfalta/git_host/agfalta_tools.git
$ git push
```

For the submodule setup, see also [this blogpost](http://blog.davidecoppola.com/2015/02/how-to-create-git-submodule-from-repository-subdirectory/).
