"""Tests the shared uspy.toml config machinery (uspy.config) and the DATADIR/LEEMDIR
path resolution built on top of it (uspy.__init__)."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import os
from pathlib import Path

import pytest

import uspy
import uspy.leem
from uspy import config


@pytest.fixture(autouse=True)
def _no_session_path_leak():
    """`set_path`/`set_datadir`/`set_leemdir` are deliberately global mutable state --
    make sure no test can leak them into another. Also strip any direct-assignment
    shadowing of `uspy.DATADIR`/`LEEMDIR` a test might have created."""
    config.reset_paths()
    uspy.__dict__.pop("DATADIR", None)
    uspy.__dict__.pop("LEEMDIR", None)
    yield
    config.reset_paths()
    uspy.__dict__.pop("DATADIR", None)
    uspy.__dict__.pop("LEEMDIR", None)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Empty HOME, empty XDG_CONFIG_HOME, empty cwd -- so a developer's real
    ~/.config/uspy/uspy.toml or ~/data can never leak into a default-value assertion."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows equivalent of HOME
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return {"home": home, "cwd": cwd}


def _write_toml(folder: Path, text: str) -> Path:
    path = folder / "uspy.toml"
    path.write_text(text)
    return path


### Defaults / back-compat


def test_default_datadir_and_leemdir(isolated_env):
    home = isolated_env["home"]
    assert uspy.DATADIR == str(home / "data") + "/"
    assert uspy.LEEMDIR == str(home / "data" / "LEEM") + "/"


def test_from_import_works():
    from uspy import DATADIR, LEEMDIR  # pylint: disable=import-outside-toplevel

    assert isinstance(DATADIR, str)
    assert isinstance(LEEMDIR, str)


def test_star_import_exports_both():
    ns: dict = {}
    exec("from uspy import *", ns)  # pylint: disable=exec-used
    assert "DATADIR" in ns
    assert "LEEMDIR" in ns


def test_values_are_strings_and_concatenable(isolated_env):
    assert isinstance(uspy.DATADIR, str)
    assert isinstance(uspy.LEEMDIR, str)
    assert uspy.LEEMDIR + "file.dat" == str(isolated_env["home"] / "data" / "LEEM") + "/file.dat"


def test_dir_lists_path_attrs():
    assert "DATADIR" in dir(uspy)
    assert "LEEMDIR" in dir(uspy)


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        uspy.NOT_A_THING  # pylint: disable=pointless-statement


def test_leem_forwards_leemdir():
    assert uspy.leem.LEEMDIR == uspy.LEEMDIR
    assert uspy.leem.DATADIR == uspy.DATADIR
    with pytest.raises(AttributeError):
        uspy.leem.NOT_A_THING  # pylint: disable=pointless-statement


### Resolution from uspy.toml


def test_cwd_config_sets_datadir(isolated_env):
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "/mnt/surfer/data"\n')
    assert uspy.DATADIR == "/mnt/surfer/data/"


def test_leemdir_derives_from_datadir(isolated_env):
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "/mnt/surfer/data"\n')
    assert uspy.LEEMDIR == "/mnt/surfer/data/LEEM/"


def test_explicit_leemdir_wins(isolated_env):
    _write_toml(
        isolated_env["cwd"],
        '[uspy.paths]\ndatadir = "/mnt/surfer/data"\nleemdir = "/mnt/surfer/other/LEEM"\n',
    )
    assert uspy.LEEMDIR == "/mnt/surfer/other/LEEM/"


def test_global_dir_config(tmp_path, monkeypatch):
    global_dir = tmp_path / "xdg" / "uspy"
    global_dir.mkdir(parents=True)
    _write_toml(global_dir, '[uspy.paths]\ndatadir = "/mnt/global/data"\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert uspy.DATADIR == "/mnt/global/data/"


def test_cwd_beats_global(tmp_path, monkeypatch, isolated_env):
    global_dir = tmp_path / "xdg" / "uspy"
    global_dir.mkdir(parents=True)
    _write_toml(global_dir, '[uspy.paths]\ndatadir = "/mnt/global/data"\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "/mnt/local/data"\n')
    assert uspy.DATADIR == "/mnt/local/data/"


def test_leemdir_follows_cwd_datadir_not_global_leemdir(tmp_path, monkeypatch, isolated_env):
    # global sets BOTH datadir and leemdir; cwd overrides only datadir. LEEMDIR must
    # move with the winning (cwd) datadir, not stay pinned to the global leemdir --
    # this is the key semantic difference from a flat per-key merge.
    global_dir = tmp_path / "xdg" / "uspy"
    global_dir.mkdir(parents=True)
    _write_toml(
        global_dir,
        '[uspy.paths]\ndatadir = "/mnt/global/data"\nleemdir = "/mnt/global/data/LEEM"\n',
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "/mnt/local/data"\n')

    assert uspy.DATADIR == "/mnt/local/data/"
    assert uspy.LEEMDIR == "/mnt/local/data/LEEM/"


def test_reader_only_config_leaves_paths_at_default(isolated_env):
    _write_toml(isolated_env["cwd"], '[uspy]\nreader = "elettra"\n')
    home = isolated_env["home"]
    assert uspy.DATADIR == str(home / "data") + "/"


### Normalization / expansion


def test_tilde_expansion(isolated_env):
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "~/mydata"\n')
    assert uspy.DATADIR == str(isolated_env["home"] / "mydata") + "/"


def test_env_var_expansion(isolated_env, monkeypatch):
    monkeypatch.setenv("MY_ROOT", "/mnt/env-root")
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "$MY_ROOT/data"\n')
    assert uspy.DATADIR == "/mnt/env-root/data/"

    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "${MY_ROOT}/data"\n')
    assert uspy.DATADIR == "/mnt/env-root/data/"


def test_unset_env_var_warns_and_keeps_literal(isolated_env, monkeypatch):
    monkeypatch.delenv("DOES_NOT_EXIST", raising=False)
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "$DOES_NOT_EXIST/data"\n')
    with pytest.warns(UserWarning):
        value = uspy.DATADIR
    assert "$DOES_NOT_EXIST" in value


@pytest.mark.parametrize(
    "raw",
    ["/mnt/data", "/mnt/data/", "/mnt/data//"],
)
def test_trailing_slash_normalization(isolated_env, raw):
    _write_toml(isolated_env["cwd"], f'[uspy.paths]\ndatadir = "{raw}"\n')
    assert Path(uspy.DATADIR) == Path("/mnt/data")
    assert uspy.DATADIR.endswith(os.sep) or uspy.DATADIR.endswith("/")
    assert not uspy.DATADIR.endswith("//")


def test_relative_value_anchored_to_config_folder(isolated_env):
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "sub/data"\n')
    expected = str(isolated_env["cwd"].resolve() / "sub" / "data") + "/"
    assert uspy.DATADIR == expected


def test_normalization_is_idempotent(isolated_env):
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "/mnt/data/"\n')
    first = uspy.DATADIR
    config.set_datadir(first)
    assert uspy.DATADIR == first


### Errors


def test_non_string_datadir_raises_typeerror(isolated_env):
    path = _write_toml(isolated_env["cwd"], "[uspy.paths]\ndatadir = 5\n")
    with pytest.raises(TypeError, match=str(path)):
        uspy.DATADIR  # pylint: disable=pointless-statement


def test_empty_datadir_raises_valueerror(isolated_env):
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = ""\n')
    with pytest.raises(ValueError):
        uspy.DATADIR  # pylint: disable=pointless-statement


def test_paths_table_not_a_table_raises_typeerror(isolated_env):
    _write_toml(isolated_env["cwd"], "[uspy]\npaths = 5\n")
    with pytest.raises(TypeError):
        uspy.DATADIR  # pylint: disable=pointless-statement


def test_malformed_toml_propagates_on_access(isolated_env):
    _write_toml(isolated_env["cwd"], "[uspy\nthis is not valid toml")
    with pytest.raises(Exception):  # tomllib.TOMLDecodeError (or tomli's) # pylint: disable=broad-exception-caught
        uspy.DATADIR  # pylint: disable=pointless-statement


def test_malformed_toml_does_not_break_import(isolated_env):
    _write_toml(isolated_env["cwd"], "[uspy\nthis is not valid toml")
    # Accessing __version__ (a plain module attribute, not a lazy one) must not touch
    # config resolution at all -- this is the whole point of resolving lazily.
    assert isinstance(uspy.__version__, str)


### Session overrides


def test_set_datadir_beats_config_and_drags_leemdir(isolated_env):
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "/mnt/from-config"\n')
    config.set_datadir("/mnt/from-session")
    assert uspy.DATADIR == "/mnt/from-session/"
    assert uspy.LEEMDIR == "/mnt/from-session/LEEM/"


def test_set_leemdir_independent_of_datadir(isolated_env):
    config.set_datadir("/mnt/data")
    config.set_leemdir("/mnt/other/LEEM")
    assert uspy.DATADIR == "/mnt/data/"
    assert uspy.LEEMDIR == "/mnt/other/LEEM/"


def test_reset_paths_restores_config_resolution(isolated_env):
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "/mnt/from-config"\n')
    config.set_datadir("/mnt/from-session")
    config.reset_paths()
    assert uspy.DATADIR == "/mnt/from-config/"
    assert config.get_path_overrides() == {}


def test_set_path_accepts_pathlike():
    config.set_datadir(Path("/mnt/pathlib-data"))
    assert uspy.DATADIR == "/mnt/pathlib-data/"


def test_direct_assignment_shadows_and_del_restores(isolated_env):
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "/mnt/from-config"\n')
    uspy.DATADIR = "/mnt/direct/"
    assert uspy.DATADIR == "/mnt/direct/"
    del uspy.DATADIR
    assert uspy.DATADIR == "/mnt/from-config/"


### Caching / freshness


def test_cache_picks_up_rewritten_file_with_new_mtime(isolated_env):
    path = _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "/mnt/first"\n')
    assert uspy.DATADIR == "/mnt/first/"

    path.write_text('[uspy.paths]\ndatadir = "/mnt/second"\n')
    # Force a distinct mtime in case the filesystem's mtime resolution is coarser than
    # the time this test takes to run.
    stat = path.stat()
    new_ns = stat.st_mtime_ns + 10**9
    os.utime(path, ns=(new_ns, new_ns))

    assert uspy.DATADIR == "/mnt/second/"


def test_search_paths_candidate_computation_is_cached_per_folder(isolated_env, monkeypatch):
    """The resolve()-based de-duplication in config_search_paths is the expensive part of
    resolving a uspy.toml (repeated .resolve() calls for the same folder); it has no
    freshness requirement of its own (unlike file content/mtimes) and should only be
    recomputed once per distinct (data folder, cwd, global config dir) combination --
    e.g. once per stack, not once per frame."""
    calls = []
    original = Path.resolve

    def counting_resolve(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counting_resolve)

    folder = isolated_env["cwd"]
    _write_toml(folder, '[uspy.paths]\ndatadir = "/mnt/first"\n')

    config.config_search_paths(str(folder / "frame_000.dat"))
    calls.clear()  # only interested in repeat calls from here on

    for i in range(1, 20):
        config.config_search_paths(str(folder / f"frame_{i:03d}.dat"))

    assert calls == []


def test_search_paths_cache_still_picks_up_content_changes(isolated_env):
    """The candidate-*path* computation being cached must not make file *content* changes
    (a rewritten uspy.toml) go stale -- that freshness still lives in the per-file
    is_file()/stat()/mtime-keyed parse, which config_search_paths' cache doesn't touch."""
    folder = isolated_env["cwd"]
    path = _write_toml(folder, '[uspy.paths]\ndatadir = "/mnt/first"\n')

    config.config_search_paths(str(folder / "frame_000.dat"))  # warm the path-list cache
    assert uspy.DATADIR == "/mnt/first/"

    path.write_text('[uspy.paths]\ndatadir = "/mnt/second"\n')
    stat = path.stat()
    new_ns = stat.st_mtime_ns + 10**9
    os.utime(path, ns=(new_ns, new_ns))

    assert uspy.DATADIR == "/mnt/second/"


def test_chdir_after_first_read_changes_datadir(tmp_path, monkeypatch, isolated_env):
    assert uspy.DATADIR == str(isolated_env["home"] / "data") + "/"

    other = tmp_path / "other_cwd"
    other.mkdir()
    _write_toml(other, '[uspy.paths]\ndatadir = "/mnt/other-cwd"\n')
    monkeypatch.chdir(other)

    assert uspy.DATADIR == "/mnt/other-cwd/"


### Extra keys


def test_extra_path_key_readable_via_get_path(isolated_env):
    _write_toml(isolated_env["cwd"], '[uspy.paths]\nxps = "~/data/XPS"\n')
    assert config.get_path("xps") == str(isolated_env["home"] / "data" / "XPS") + "/"
    with pytest.raises(AttributeError):
        uspy.XPSDIR  # pylint: disable=pointless-statement


def test_get_path_unknown_key_returns_none_or_default(isolated_env):
    assert config.get_path("nope") is None
    assert config.get_path("nope", "~/fallback") == str(isolated_env["home"] / "fallback") + "/"


def test_resolve_paths_introspection(isolated_env):
    _write_toml(isolated_env["cwd"], '[uspy.paths]\ndatadir = "/mnt/data"\n')
    assert config.resolve_paths() == {"datadir": "/mnt/data/", "leemdir": "/mnt/data/LEEM/"}
