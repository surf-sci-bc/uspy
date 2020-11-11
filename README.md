# Setting up agfalta_tools

If you just want to work with this on your local computer, just follow the first two sections [Get source](#Get-source) and [Installation](Installation).

## Get source

Simply clone the github repo (it is a private repo, so you need access). Use either the first or the second line. For the second, you need [ssh access to git](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/connecting-to-github-with-ssh).

```sh
$ git clone https://github.com/surf-sci-bc/agfalta_tools.git
$ git clone git@github.com:surf-sci-bc/agfalta_tools.git
```

### Pushing to the jupyterlab production server

*skip this if you only want to install locally*

If you want to push to the jupyterlab production server in the IFP group at Uni Bremen, you need to add this as a separate remote (call it `deployment`). You also need ssh access to the machine and a user that is in the `githost` group so you have write access to the server. 

```sh
$ git remote add deployment user@192.168.2.63:/home/agfalta/git_host/agfalta_tools.git
```

Now, when you do `git pull` you will get your files from github. For pushing, you can do either `git push` to push to github or `git push deployment` to push to the IFP server.


## Installation

##### Create venv

To be able to run your code, you have to set up a python3 virtual environment. On Linux, do this:

```sh
$ cd path/to/agfalta_tools
$ python3 -m venv venv
```

On Windows, you can set up a venv in PyCharm. For anything where you want to run agfalta_tools code, you need to make sure you are in the venv (see next paragraph). In the Terminal, it says "(venv)" at the beginning of each line if you are in the venv.

##### Enter venv

On Linux, do

```sh
$ cd path/to/agfalta_tools
$ source venv/bin/activate
```

On Windows, this instead:

```cmd
venv\Scripts\activate
```

##### Installation

Sometimes, the pip version of your new virtual environment will be too old to properly install opencv. To prevent this, do:

```sh
(venv) $ python3 -m pip install --upgrade pip
```

And finally, install your local `agfalta` module in editable mode:

```sh
(venv) $ python3 -m pip install -e .
```

Now everything should run and you can do `import agfalta` from your virtual environment.


## Testing

The project and its `testdata` folder are two separate repositories because all the binary data in `testdata` makes the pushing and pulling too slow otherwise. If you don't do tests, you don't need `testdata`. If you do, do this in the agfalta_tools repo:

```sh
$ git submodule init
$ git submodule update
```

For running the tests, just go into `agfalta_tools` venv and do

```sh
(venv) $ pytest
```


## Set up the git repos on the server

*NOTE: Maybe it is easier to just clone the repo from github instead of creating a bare repo and pushing to it?*

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
