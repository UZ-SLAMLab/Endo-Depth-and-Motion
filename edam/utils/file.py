from pathlib import Path
from typing import List, Optional


def list_files(directory: str, pattern: Optional[str] = None) -> List[str]:
    """List of files in a directory.

    Example:
    # Given a directory tree:
    # test
    # ├── file1
    # ├── file2
    # ├── file3
    # └── file4.txt
    >>> list_files("test")
    ['test/file4.txt', 'test/file1', 'test/file2', 'test/file3']
    >>> list_files("test/")
    ['test/file4.txt', 'test/file1', 'test/file2', 'test/file3']
    >>> list_files("test/","*.txt")
    ['test/file4.txt']

    Arguments:
        directory {str} -- Path to the directory.

    Keyword Arguments:
        pattern {Optional[str]} -- Only files that match the pattern will be listed.
            (default: {None})

    Returns:
        List[str] -- Return a list of paths in str format.
    """
    pattern = pattern or "*"
    return list(map(str, Path(directory).glob(pattern)))


def list_files_recursive(directory: str, pattern: Optional[str] = None) -> List[str]:
    """List of files in a directory recursively.

    Example:
    # Given a directory tree:
    # test
    # ├── dir1
    # │   ├── file5
    # │   └── file6.txt
    # ├── file1
    # ├── file2
    # ├── file3
    # └── file4.txt
    >>> list_files_recursive("test/")
    ['test/file4.txt', 'test/dir1', 'test/file1', 'test/file2', 'test/file3', 'test/dir1/file5', 'test/dir1/file6.txt'] # noqa: E501
    >>> list_files_recursive("test/","*.txt")
    ['test/file4.txt', 'test/dir1/file6.txt']
    >>> list_files_recursive("test/dir","*.txt")
    []
    >>> list_files_recursive("test/dir1","*.txt")
    ['test/dir1/file6.txt']

    Arguments:
        directory {str} -- Path to the directory.

    Keyword Arguments:
        pattern {Optional[str]} -- Only files that match the pattern will be listed.
            (default: {None})

    Returns:
        List[str] -- Return a list of paths in str format.
    """
    pattern = pattern or "*"
    return list(map(str, Path(directory).rglob(pattern)))


def path_remove_prefix(
    path: str, prefix: str, ignore_error: Optional[bool] = None
) -> str:
    """Remove a prefix from a path.

    Example:
    >>> path_remove_prefix("/home/foo/Downloads","")
    ValueError: '/home/foo/Downloads' does not start with ''
    >>> path_remove_prefix("/home/foo/Downloads","/")
    'home/foo/Downloads'
    >>> path_remove_prefix("/home/foo/Downloads","/home")
    'foo/Downloads'
    >>> path_remove_prefix("/home/foo/Downloads","/home/")
    'foo/Downloads'
    >>> path_remove_prefix("/home/foo/Downloads","/home/foo")
    'Downloads'
    >>> path_remove_prefix("/home/foo/Downloads","/home/fo")
    ValueError: '/home/foo/Downloads' does not start with '/home/fo'
    >>> path_remove_prefix("/home/foo/Downloads","/home/fo", ignore_error = True)
    '/home/foo/Downloads'

    Arguments:
        file_path {str} -- Path of the file.
        prefix {str} -- Prefix to be removed.

    Keyword Arguments:
        ignore_error {Optional[bool]} -- Ignore error, if the `file_path` does not start
            with the prefix, the same `file_path` will be returned. (default: False)

    Raises:
        ValueError: If path does not starts with prefix.

    Returns:
        str -- Resulting path
    """
    ignore_error = ignore_error or False
    try:
        return str(Path(path).relative_to(prefix))
    except ValueError as e:
        if ignore_error:
            return path
        else:
            raise ValueError(e)


def list_path_remove_prefix(
    path_list: List[str], prefix: str, ignore_error: Optional[bool] = None
) -> List[str]:
    """Remove path prefix to a list of paths.

    See `path_remove_prefix` for more details.

    Arguments:
        path_list {List[str]} -- List of paths.
         prefix {str} -- Prefix to be removed.

    Keyword Arguments:
        ignore_error {Optional[bool]} -- Ignore error, if the `file_path` does not start
            with the prefix, the same `file_path` will be returned. (default: False)

    Raises:
        ValueError: If path does not starts with prefix.

    Returns:
        List[str] -- List of resulting paths.
    """
    return list(
        map(
            lambda x: path_remove_prefix(x, prefix, ignore_error=ignore_error),
            path_list,
        )
    )


def order_list_paths_by_int_filename(list_paths: List[str]) -> List[str]:
    """Order a list of paths by the filenames. It assumes that the file names
    are integer. E.g. "dir1/dir_b/001.png"

    Example:
    >>> order_list_paths_by_int_filename(["a/001.png", "b/000.png", "2.png"])
    ["b/000.png", "a/001.png", "2.png"]
    >>> order_list_paths_by_int_filename(["b/001.png", "2.png", "a/000.png"])
    ["a/000.png", "b/001.png", "2.png"]

    Arguments:
        list_paths {List[str]} -- List of paths

    Returns:
        List[str] -- Same list of paths in ascendent order.
    """
    list_names_as_int = list(
        map(lambda x: int(x.split("/")[-1].split(".")[0]), list_paths)
    )
    _, list_paths = zip(*sorted(zip(list_names_as_int, list_paths)))  # type: ignore
    return list_paths
