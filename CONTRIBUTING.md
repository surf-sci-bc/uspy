# How to contribute to agfalta_tools

## Creating a working copy of the module

First, you need to clone the git repository to your local machine. You need to
have git and ssh installed and you need to have a user on the jupyterlab
machine that is in the `githost` group.

The project and its `testdata` folder are two separate repositories because
all the binary data in `testdata` makes the pushing and pulling too slow
otherwise. If you don't do tests, you don't need `testdata`.

Clone the source code into your projects folder:

```sh
$ cd path/to/projects
$ git clone username@192.168.2.63:/home/agfalta/git_host/agfalta_tools.git
```

Now you have a working copy and can alter the code, do commits and push them
(if you have access to the AG LAN). This is enough to start coding if you don't
want to run the code yourself (for example if you tested it on jupyterlab
already)


## Set up a virtual environment

To be able to run your code, you have to set up a python3 virtual
environment. On Linux, do this:

```sh
$ cd path/to/agfalta_tools
$ python3 -m venv venv
```

On Windows, you can set up a venv in PyCharm. For the following, you need to
make sure to be in the venv (in the Terminal, it says "(venv)" at the
beginning of each line). On Linux, do

```sh
$ cd path/to/agfalta_tools
$ source venv/bin/activate
```

On Windows, this instead:

```cmd
venv\Scripts\activate
```

Now, some dependencies are needed because pyclustering does not them declare
them correctly (matplotlib):

```sh
(venv) $ python3 -m pip install --upgrade pip        			# sometimes necessary
(venv) $ python3 -m pip install matplotlib
```

And finally:

```sh
(venv) $ python3 -m pip install -e .               	      # install agfalta module
```

Now everything should run and you can do `import agfalta` from your virtual
environment.


## Testing

You need the additional `testdata` submodule for this:

```sh
$ cd path/to/agfalta_tools
$ git submodule init
$ git submodule add username@192.168.2.63:/home/agfalta/git_host/agfalta_tools_testdata.git
```

For running the tests, just go into `agfalta_tools` and do

```sh
$ cd path/to/agfalta_tools
$ source venv/bin/activate
(venv) $ pytest
```


## Set up the git repos on the server

See also: https://gist.github.com/noelboss/3fe13927025b89757f8fb12e9066f2fa

If you need to set up the server again, you need to keep an up-to-date working
copy on some machine. Then, create an empty repo in `/home/agfalta/git_host`
(this location only matters here and for cloning afterwards, see top of this
document):

```sh
$ ssh agfalta@192.168.2.63
$ cd /home/agfalta/
$ mkdir git_host && cd git_host
$ git init --bare agfalta_tools.git
$ git init --bare agfalta_tools_testdata.git
$ sudo chown -R agfalta:githost .
```

Then, copy the `post-receive` script from your local working copy
`agfalta_tools/deployment/` into `agfalta_tools.git/hooks` and remember to
make it executable:

```sh
$ chmod +x post-receive
```

From your local machine, add the bare server repo as origin remote:

```sh
$ cd /path/to/agfalta_tools
$ git remote add origin username@192.168.2.63:/home/agfalta/git_host/agfalta_tools.git
$ git push
$ cd testdata
$ git remote add origin username@192.168.2.63:/home/agfalta/git_host/agfalta_tools_testdata.git
$ git push
```

For the submodule setup, see also http://blog.davidecoppola.com/2015/02/how-to-create-git-submodule-from-repository-subdirectory/.
