# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/biological-alignment-benchmarks/bioblue

import os
import sys
import codecs
import time
import datetime
import io
import pickle
import gzip
from pathlib import Path
import csv
import re


sentinel = object() # https://web.archive.org/web/20200221224620id_/http://effbot.org/zone/default-values.htm


is_dev_machine = (os.name == 'nt')
debugging = (is_dev_machine and sys.gettrace() is not None) and (1 == 1)  # debugging switches



data_dir = "data"

# NB! Under Windows need to prepend \\?\ in order to be able to create long filenames for cache files
if os.name == 'nt':
  # data_dir = "\\\\?\\" + os.path.abspath(data_dir)
  data_dir = os.path.abspath(data_dir)

if not os.path.exists(data_dir):
  os.makedirs(data_dir)



def safeprint(text = ""):

  text = str(text).encode("utf-8", 'ignore').decode('ascii', 'ignore')
  print(text)

#/ def safeprint(text):


def get_now_str():
  now_str = datetime.datetime.strftime(datetime.datetime.now(), "%m.%d %H:%M:%S")
  return now_str


# https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
class Timer(object):
  def __init__(self, name=None, quiet=False):
    self.name = name
    self.quiet = quiet

  def __enter__(self):
    if not self.quiet and self.name:
      safeprint(get_now_str() + " : " + self.name + "...")

    self.tstart = time.time()

  def __exit__(self, type, value, traceback):
    elapsed = time.time() - self.tstart

    if not self.quiet:
      if self.name:
        safeprint(
          get_now_str() + " : " + self.name + " totaltime: {}".format(elapsed)
        )
      else:
        safeprint(get_now_str() + " : " + "totaltime: {}".format(elapsed))
    # / if not quiet:

# / class Timer(object):


def wait_for_enter(message=None):
  if os.name == "nt":
    import msvcrt

    if message is not None:
      print(message)
    msvcrt.getch()  # Uses less CPU on Windows than input() function. This becomes perceptible when multiple console windows with Python are waiting for input. Note that the graph window will be frozen, but will still show graphs.
  else:
    if message is None:
      message = ""
    input(message)


def rename_temp_file(filename, make_backup = False):  # NB! make_backup is false by default since this operation would not be atomic

  max_tries = 20
  try_index = 1
  while True:

    try:

      if make_backup and os.path.exists(filename):

        if os.name == 'nt':   # rename is not atomic on windows and is unable to overwrite existing file. On UNIX there is no such problem
          if os.path.exists(filename + ".old"):
            if not os.path.isfile(filename + ".old"):
              raise ValueError("" + filename + ".old" + " is not a file")
            os.remove(filename + ".old")

        os.rename(filename, filename + ".old")

      #/ if make_backup and os.path.exists(filename):


      if os.name == 'nt':   # rename is not atomic on windows and is unable to overwrite existing file. On UNIX there is no such problem
        if os.path.exists(filename):
          if not os.path.isfile(filename):
            raise ValueError("" + filename + " is not a file")
          os.remove(filename)

      os.rename(filename + ".tmp", filename)

      return

    except Exception as ex:

      if try_index >= max_tries:
        raise

      try_index += 1
      safeprint("retrying temp file rename: " + filename)
      sleep(5)
      continue

    #/ try:

  #/ while True:

#/ def rename_temp_file(filename):


def read_file(filename, default_data = sentinel, quiet = False):
  """Reads a pickled file"""

  # https://web.archive.org/web/20200221224620id_/http://effbot.org/zone/default-values.htm
  if default_data is sentinel:
    default_data = {}

  fullfilename = os.path.join(data_dir, filename)

  if not os.path.exists(fullfilename + ".gz"):
    return default_data

  with Timer("file reading : " + filename, quiet):

    try:
      with open(fullfilename + ".gz", 'rb', 1024 * 1024) as fh:
        compressed_data = fh.read()    # TODO: decompress directly during reading and without using intermediate buffer for async data
        with io.BytesIO(compressed_data) as bfh:   
          with gzip.open(bfh, 'rb') as gzip_file:
            data = pickle.load(gzip_file)
    except FileNotFoundError:
      data = default_data

  #/ with Timer("file reading : " + filename):

  return data

#/ def read_file(filename):


def save_file(filename, data, quiet = False, make_backup = False):
  """Writes to a pickled file"""

  haslen = hasattr(data, '__len__')
  message_template = "file saving {}" + (" num of all entries: {}" if haslen else "")
  message = message_template.format(filename, len(data) if haslen else 0)

  with Timer(message, quiet):

    fullfilename = os.path.join(data_dir, filename)

    with open(fullfilename + ".gz.tmp", 'wb', 1024 * 1024) as fh:
      with gzip.GzipFile(fileobj=fh, filename=filename, mode='wb', compresslevel=compresslevel) as gzip_file:
        pickle.dump(data, gzip_file)
        gzip_file.flush() # NB! necessary to prevent broken gz archives on random occasions (does not depend on input data)
      fh.flush()  # just in case

    rename_temp_file(fullfilename + ".gz", make_backup)

  #/ with Timer("file saving {}, num of all entries: {}".format(filename, len(cache))):

#/ def save_file(filename, data):


def save_txt(filename, str, quiet = False, make_backup = False, append = False, use_bom = True, encoding="utf-8"):
  """Writes to a text file"""

  message_template = "file saving {} num of characters: {}"
  message = message_template.format(filename, len(str))

  with Timer(message, quiet):

    fullfilename = os.path.join(data_dir, filename)

    with open(fullfilename + ("" if append else ".tmp"), 'at' if append else 'wt', 1024 * 1024, encoding=encoding) as fh:    # wt format automatically handles line breaks depending on the current OS type
      if use_bom:
        # fh.write(codecs.BOM_UTF8 + str.encode("utf-8", "ignore"))
        fh.write(codecs.BOM_UTF8.decode("utf-8"))    # TODO: encoding
      fh.write(str)
      fh.flush()  # just in case

  #/ with Timer("file saving {}, num of all entries: {}".format(filename, len(cache))):

  if not append:
    rename_temp_file(fullfilename, make_backup)

#/ def save_txt(filename, data):


class EventLog(object):
  default_gzip_compresslevel = 6  # 6 is default level for gzip: https://linux.die.net/man/1/gzip and https://github.com/ebiggers/libdeflate

  def __init__(
    self,
    experiment_dir,
    events_fname,
    headers,
    gzip_log=False,
    gzip_compresslevel=None,
  ):
    record_path = Path(os.path.join(experiment_dir, events_fname))
    # logger.info(f"Saving records to disk at {record_path}")
    record_path.parent.mkdir(exist_ok=True, parents=True)

    if isinstance(headers, dict):
      self.header_keys = list(headers.keys())  # used with log_event_from_dict
      headers = list(headers.values())
    else:
      self.header_keys = headers

    if gzip_log:
      if gzip_compresslevel is None:
        gzip_compresslevel = self.default_gzip_compresslevel
      write_header = not os.path.exists(record_path + ".gz")
      self.file = gzip.open(
        record_path + ".gz",
        mode="at",
        newline="",
        encoding="utf-8",
        compresslevel=gzip_compresslevel,
      )  # csv writer creates its own newlines therefore need to set newline to empty string here   # TODO: buffering for gzip
    else:
      write_header = not os.path.exists(record_path)
      self.file = open(
        record_path,
        mode="at",
        buffering=1024 * 1024,
        newline="",
        encoding="utf-8",
      )  # csv writer creates its own newlines therefore need to set newline to empty string here

    self.writer = csv.writer(self.file, quoting=csv.QUOTE_MINIMAL, delimiter="\t")

    if (
      write_header
    ):  # TODO: if the file already exists then assert that the header is same
      self.writer.writerow(headers)
      # self.file.flush()

  def log_event(self, event):

    if isinstance(event, dict):
      values = [event.get(key) for key in self.header_keys]
    else:
      values = event

    # transformed_cols = []
    # for index, col in enumerate(event):
    #   # if type(col) == datetime.datetime:
    #   #  col = datetime.datetime.strftime(col, '%Y.%m.%d-%H.%M.%S')
    #   transformed_cols.append(col)

    values = [
      x.strip().replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")   # CSV/TSV format does not support these characters
      # re.sub(r"[\n\r\t]", " ", x.strip())   # CSV/TSV format does not support these characters
      if isinstance(x, str) 
      else x 
      for x in values
    ]

    self.writer.writerow(values)
    # self.file.flush()

  def flush(self):
    self.file.flush()

  def close(self):
    self.file.flush()
    self.file.close()

# / class EventLog(object):
