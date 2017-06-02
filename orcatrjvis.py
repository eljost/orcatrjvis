#!/usr/bin/env python3

import argparse
from collections import namedtuple
import glob
import logging
import os
import subprocess
import sys

from jinja2 import Template
import yaml

from qchelper.paths import search_files_with_ext
from qchelper.parser.Orca import Orca


JMOL_TPL = Template("""load trajectory {{ trj_fn }}
set frank off
frame 1
num_frames = getProperty("modelInfo.modelCount")
for (var i = 1; i <= num_frames; i = i+1)
    var filename = "{{ base_out_fn }}"+("00000"+i)[-4][0]+".png"
    write IMAGE 800 600 PNG @filename
    frame next
end for
""")
REPORT_TPL = Template("""<!doctype html>
<html>
    <head>
        <meta charset="utf-8"/>
        <title>Imaginary vibrations</title>
    </head>
    <body>
    {% for fn in imgvib_dict %}
    <div>
        <h1>{{ fn }}</h1>
        {% for iv in imgvib_dict[fn] %}
        <div style="display: inline-block;">
            <h2>Index: {{ iv.index }}: {{ iv.value }} cm<sup>-1</sup></h2>
            <img src="{{ iv.movie }}" with="400" height="300" alt="{{ iv.movie }}"></img>
        </div>
        {% endfor %}
    </div>
    {% endfor %}
    </body>
</html>
""")

ImgVib = namedtuple("ImgVib", "fn index value movie")


def parse_args(args):
    parser = argparse.ArgumentParser("#Fun")
    parser.add_argument("root_dir")

    return parser.parse_args(args)


def video_from_trajectory(trj_fn, trj_ind):
    head, tail = os.path.split(trj_fn)

    trj_dir = os.path.join(head, "trj{}".format(trj_ind))
    try:
        os.mkdir(trj_dir)
    except FileExistsError:
        pass
    base_out_fn = os.path.join(trj_dir, "movie")
    jmol_script = JMOL_TPL.render(trj_fn=trj_fn, base_out_fn=base_out_fn)
    jmol_script_fn = os.path.join(trj_dir, "animate.spt")
    with open(jmol_script_fn, "w") as handle:
        handle.write(jmol_script)
    # -n: no display
    jmol_cmd = "jmol -n {}".format(jmol_script_fn).split()
    subprocess.call(jmol_cmd)
    image_fns = " ".join(glob.glob(os.path.join(trj_dir, "*.png")))
    movie_fn = os.path.join(trj_dir, "trj{}.gif".format(trj_ind))
    movie_cmd = "convert {} {}".format(image_fns, movie_fn).split()
    subprocess.call(movie_cmd)

    return movie_fn


def create_report(imgvib_dict):
    report = REPORT_TPL.render(imgvib_dict=imgvib_dict)
    with open("report.html", "w") as handle:
        handle.write(report)


def run():
    args = parse_args(sys.argv[1:])

    log_paths = search_files_with_ext(args.root_dir, ext=".out",
                                      ignore_fns=("slurm", ))
    imgvib_dict = dict()
    for orca_log_fn in log_paths:
        imgvib_dict[orca_log_fn] = vibs_from_orca_log(orca_log_fn)

    # Save ImgVib namedtuples to a file
    with open("imgvibs.yaml", "w") as handle:
        handle.write(yaml.dump(imgvib_dict))

    create_report(imgvib_dict)


def vibs_from_orca_log(orca_log_fn):
    """
    Assuminig a non-linear molecule where the first imaginary frequency
    appears at index 6!.
    """
    orca_parser = Orca(orca_log_fn)
    index_range = range(6, 6+len(orca_parser.imgvibfreqs))
    str_inds = [str(ind) for ind in index_range]
    cmd = "orca_pltvib {} {}".format(orca_parser.fn, " ".join(str_inds)).split()
    subprocess.call(cmd)
    trj_fns = ["{}.v{}.xyz".format(orca_parser.fn, i.zfill(3)) for i in str_inds]

    movie_fns = [video_from_trajectory(trj_fn, trj_ind)
                 for trj_fn, trj_ind in zip(trj_fns, index_range)]

    imgvibs = [ImgVib(fn=orca_parser.fn, 
                      index=index,
                      value=float(value),
                      movie=movie)
                for index, value, movie
                in zip(index_range, orca_parser.imgvibfreqs, movie_fns)
    ]
    return imgvibs

if __name__ == "__main__":
    run()
