# agfalta_tools

If you just want to work with this on your local computer, just follow the first two sections [Get source](#Get-source) and [Installation](Installation). 

For deployment information, refer to [this](deployment/README.md).

## Get source

Simply clone the github repo (it is a private repo, so you need access). Use either the first or the second line. For the second, you need [ssh access to git](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/connecting-to-github-with-ssh).

```sh
$ git clone https://github.com/surf-sci-bc/agfalta_tools.git
$ git clone git@github.com:surf-sci-bc/agfalta_tools.git
```

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

## Contributing

### Testing

The project and its `testdata` folder are two separate repositories because all the binary data in `testdata` makes the pushing and pulling too slow otherwise. If you don't do tests, you don't need `testdata`. If you do, do this in the agfalta_tools repo:

```sh
$ git submodule init
$ git submodule update
```

For running the tests, just go into `agfalta_tools` venv and do

```sh
(venv) $ pytest
```

### Versioning

Versioning is managed by `setuptools_scm`, which uses the current git tag to determine the version. The syntax of the version identifier is  [Semantic Versioning](https://semver.org/) and therefore follows the classic `MAJOR.MINOR.PATCH` scheme. You can tag the **last commit** as version `{x}.{y}.{z}` with the following command:

```sh
$ git tag -a "{x}.{y}.{z}" -m "Version description"
```

Check the current tag via `$ git describe`. If the current commit has not been tagged, the version is called `{x}.{y}.{z+1}.dev{d}+g{commit hash}.d{date}` where the last part is only present if the repo is dirty (= uncommitted changes).

Remember that after tagging a commit, you have to push the tags in addition to the normal commits, so for pushing both to all, do:

```sh
$ git push all
$ git push all --tags
```

You can get the current version by `from agfalta.version import __version__`. This will either retrieve the version from git if `setuptools_scm` is installed and the install lives in a git repository. Otherwise, it will look in the package metadata which are from installation time and might thus be outdated on editable installs.