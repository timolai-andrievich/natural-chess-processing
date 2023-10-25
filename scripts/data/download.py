#!python
"""
A script for downloading the raw data.
"""
import argparse
from dataclasses import dataclass
import os
from typing import List
import urllib.parse
import warnings

import pyzstd
import requests
import tqdm

# 121,332 games, 18 MB
URL_SMALL = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
FILE_SMALL = 'small.pgn.zst'

# 1,048,440 games, 200 MB
URL_MEDIUM = "https://database.lichess.org/standard/lichess_db_standard_rated_2014-07.pgn.zst"
FILE_MEDIUM = 'medium.pgn.zst'

# 101,706,224 games, 33 GB
URL_LARGE = "https://database.lichess.org/standard/lichess_db_standard_rated_2023-04.pgn.zst"
FILE_LARGE = 'large.pgn.zst'

REPOSITORY_NAME = 'natural-chess-processing'


@dataclass
class ArgsTuple:
    """Wrapper class for program arguments, needed for typing.
    """
    url: str
    quiet: bool
    output_directory: str
    no_guessing: bool
    unzip: bool
    filename: str


def parse_args() -> ArgsTuple:
    """Parses arguments passed to the function and returns them.

    Returns:
        ArgsTuple: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        'download.py', "Downloads the text-detoxification dataset")
    parser.add_argument('--url',
                        type=str,
                        default=URL_SMALL,
                        help='The url of the .zip file containing the dataset')
    parser.add_argument(
        '--small',
        action='store_const',
        dest='url',
        const=URL_SMALL,
        help="Download small database - 127K games, 18 MB compressed",
    )
    parser.add_argument(
        '--medium',
        action='store_const',
        dest='url',
        const=URL_MEDIUM,
        help="Download medium database - 1M games, 200 MB compressed",
    )
    parser.add_argument(
        '--large',
        action='store_const',
        dest='url',
        const=URL_LARGE,
        help="Download large database - 30M games, 30 GB compressed",
    )
    parser.add_argument('--quiet',
                        action='store_true',
                        help='Disable progress bars.')
    parser.add_argument('--output-directory',
                        default='./',
                        help='Where to put the dataset.')
    parser.add_argument(
        '--no-guessing',
        action='store_true',
        help=
        'Do not try to find the /data directory, ' + \
        'and download directly into the output directory.'
    )
    parser.add_argument('--no-unzip',
                        action='store_false',
                        dest='unzip',
                        help='Do not unzip the downloaded data automatically')
    parser.add_argument('--filename',
                        type=str,
                        default=None,
                        help='The name of the downloaded file. '
                        'Inferred from the download url by default.')
    args: ArgsTuple = parser.parse_args()
    if args.filename is None:
        if args.url == URL_SMALL:
            args.filename = FILE_SMALL
        elif args.url == URL_MEDIUM:
            args.filename = FILE_MEDIUM
        elif args.url == URL_LARGE:
            args.filename = FILE_LARGE
        else:
            path = urllib.parse.urlparse(args.url).path
            basename = os.path.basename(path)
            args.filename = basename
    return args


def get_path_directories(path: os.PathLike) -> List[str]:
    """Returns the list of names of all directories that
    are either ancestors of `path`, including the name of `path`, if `path` is a folder.

    Args:
        path (os.PathLike): The path to be parsed.

    Returns:
        List[str]: List of names of directories leading up to `path`, including `path` itself.
    """
    ancestor_directories = []
    while True:
        path, directory = os.path.split(path)
        if directory:
            ancestor_directories.append(directory)
        else:
            ancestor_directories.append(path)
            break
    ancestor_directories.reverse()
    return ancestor_directories


def get_subdirectories(path: str) -> List[str]:
    """Returns the list of names of directories that are in the directory,
    specified for `path`

    Args:
        path (str): The path to the target directory.

    Returns:
        List[str]: The list of names of directories that are in the directory.
    """
    _top, subdirectories, _files = next(os.walk(path))
    return subdirectories


def guess_data_directory(target_directory: str) -> str:
    """Tries to find the /data directory, where / is the root of the repository.

    Args:
        target_directory (str): The directory from where script was called.

    Returns:
        str: The absolute path to the `/data` directory
    """
    target_directory = os.path.abspath(target_directory)

    # Check if the target directory exists
    if not os.path.exists(target_directory):
        raise FileNotFoundError(f'Directory {target_directory} not found.')
    if not os.path.isdir(target_directory):
        raise FileExistsError(f'{target_directory} is not a directory.')

    # Attempt to get to the root repository directory
    path_directories = get_path_directories(target_directory)
    if path_directories and path_directories[-2:] == ['scripts', 'data']:
        # Check if the script is run from the same directory it is in.
        target_directory = os.path.join(*path_directories[:-2])  # pylint: disable=no-value-for-parameter
    elif path_directories and path_directories[-1] == 'scripts':
        # Check if the script is run from the scripts directory
        target_directory = os.path.join(*path_directories[:-1])  # pylint: disable=no-value-for-parameter
    elif REPOSITORY_NAME not in path_directories:
        # If the name of the repository is not in the path, check children directories
        subdirectories = get_subdirectories(target_directory)
        if REPOSITORY_NAME in subdirectories:
            target_directory = os.path.join(target_directory, REPOSITORY_NAME)

    if REPOSITORY_NAME in path_directories:
        # If the target directory is the root of the repository, create ./data/ directory
        # if it doesn't exist, and set it as a target directory
        data_dir = os.path.join(target_directory, 'data')
        if os.path.exists(data_dir) and not os.path.isdir(data_dir):
            raise FileExistsError(f'{data_dir} is not a directory.')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        target_directory = data_dir

    return target_directory


def download_file(target_directory: str, url: str, quiet: bool,
                  filename: str) -> str:
    """Downloads the file to the target_directory.

    Args:
        target_directory (str): The directory to download the file to.
        url (str): The url of the file.
        quiet (bool): Whether to hide the progress bar or not.

    Returns:
        str: The name of the downloaded file.
    """
    filename = os.path.join(target_directory, filename)
    with requests.get(url, stream=True,
                      timeout=10) as request, open(filename, 'wb') as file:
        request.raise_for_status()
        total_size = int(request.headers.get('Content-Length', 0))
        chunk_size = 16 * 2**10
        if total_size == 0:
            bar_format = '{elapsed} {rate_noinv_fmt}'
        else:
            bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n:.3f}{unit}/{total:.3f}{unit} ' +\
                '[{elapsed}<{remaining}, {rate_noinv_fmt}]'
        with tqdm.tqdm(total=total_size / 2**20,
                       disable=quiet,
                       unit='MB',
                       desc='Downloading',
                       bar_format=bar_format) as pbar:
            for chunk in request.iter_content(chunk_size):
                file.write(chunk)
                pbar.update(len(chunk) / 2**20)
    return filename


def unzip(file_path: str, target_directory: str, quiet: bool):
    """Unzips the downloaded file.

    Args:
        file_path (str): The path to the downloaded .zip file.
        target_directory (str): The directory to unzip the file to.
    """
    basename = os.path.basename(file_path)
    target_file = os.path.join(target_directory, basename)
    if not file_path.endswith('.zst'):
        target_file = target_file + '.decompressed'
        warnings.warn(
            f'The file {file_path} does not have the zstd file extension')
    else:
        target_file = target_file[:-4]
    if not quiet:
        print(f'Unzipping {file_path} -> {target_file}')
    total_size = os.path.getsize(file_path) / 2**20
    bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n:.3f}{unit}/{total:.3f}{unit} ' +\
        '[{elapsed}<{remaining}, {rate_noinv_fmt}] {postfix}'
    with open(file_path, 'rb') as fin, open(
            target_file, 'wb') as fout, tqdm.tqdm(total=total_size,
                                                  bar_format=bar_format,
                                                  unit='MB',
                                                  disable=quiet,
                                                  desc='Decompressing') as pbar:
        read_size = 0
        def update_pbar(total_input, total_output, _read_data, _write_data):
            nonlocal read_size
            delta = total_input - read_size
            read_size = total_input
            pbar.update(delta / 2**20)
            pbar.set_postfix_str(f'Decompressed size: {total_output/2**20:.3f}MB', refresh=False)
        pyzstd.decompress_stream(fin, fout, callback=update_pbar) # pylint: disable=no-member
        pbar.refresh()


def main():
    """The main function of the program.
    """
    args = parse_args()
    if args.no_guessing:
        target_directory = args.output_directory
    else:
        target_directory = guess_data_directory(args.output_directory)
    if not args.quiet:
        print(f'Downloading into directory {target_directory}.')
    downloaded_file = download_file(target_directory, args.url, args.quiet,
                                    args.filename)
    if args.unzip:
        unzip(downloaded_file, target_directory, args.quiet)
        os.remove(downloaded_file)


if __name__ == '__main__':
    main()
