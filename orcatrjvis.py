#!/usr/bin/env python3

import argparse
from collections import namedtuple, OrderedDict
import glob
import logging
import os
import re
import subprocess
import sys

from jinja2 import Template
from natsort import natsorted
import numpy as np
import yaml

from qchelper.paths import search_files_with_ext
from qchelper.parser.Orca import Orca


JMOL_TPL = Template("""load trajectory {{ trj_fn }}
set frank off
frame 1
num_frames = getProperty("modelInfo.modelCount")
for (var i = 1; i <= num_frames; i = i+1)
    var filename = "{{ base_out_fn }}"+("00000"+i)[-4][0]+".png"
    write IMAGE 1024 768 PNG @filename
    frame next
end for
""")
REPORT_BASE = """<!doctype html>
<html>
    <head>
        <meta charset="utf-8"/>
        <title>{}</title>
    </head>
    <body>
    {}
    </body>
</html>

"""
IMGVIB_REPORT_TPL = Template(REPORT_BASE.format("Imaginary vibrations",
    """
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
    {% endfor %}""")
)
TRJ_REPORT_TPL = Template(REPORT_BASE.format("Trajectories",
    """
    {% for trj_fn, movie_fn in trj_movie_zipped %}
    <div>
        <h1>{{ trj_fn }}</h1>
        <img src="{{ movie_fn }}" with="600" height="600" alt="{{ movie_fn }}"></img>
    </div>
    {% endfor %}
    """)
)
ImgVib = namedtuple("ImgVib", "fn index value movie")


def parse_args(args):
    parser = argparse.ArgumentParser("Visualize imaginary frequencies and "
                                     "trjactories from ORCA runs.")
    parser.add_argument("root_dir")
    vis_type = parser.add_mutually_exclusive_group(required=True)
    vis_type.add_argument("--imgvib", action="store_true",
                          help="Visualize imaginary vibrations.")
    vis_type.add_argument("--trj", action="store_true",
                          help="Visualize .trj files")
    vis_type.add_argument("--hess", action="store_true",
                          help="Visualize (multiple) hessian(s) from a dir.")

    return parser.parse_args(args)


def movie_from_trajectory(trj_fn, movie_base_fn, trj_ind="", trj_dir_suf=""):
    head, tail = os.path.split(trj_fn)

    trj_dir = os.path.join(head, "{}{}{}".format(movie_base_fn,
                                                 trj_dir_suf,
                                                 trj_ind))
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
    movie_fn = os.path.join(trj_dir, "{}{}.gif".format(movie_base_fn, trj_ind))
    movie_cmd = "convert {} {}".format(image_fns, movie_fn).split()
    subprocess.call(movie_cmd)

    return movie_fn


def create_imgvib_report(imgvib_dict):
    natsorted_keys = natsorted(imgvib_dict.keys())
    sorted_dict = OrderedDict()
    for key in natsorted_keys:
        sorted_dict[key] = imgvib_dict[key]
    report = IMGVIB_REPORT_TPL.render(imgvib_dict=sorted_dict)
    with open("imgvib_report.html", "w") as handle:
        handle.write(report)


def create_trj_report(trj_fns, movie_fns):
    trj_movie_zipped = zip(trj_fns, movie_fns)
    report = TRJ_REPORT_TPL.render(trj_movie_zipped=trj_movie_zipped)
    with open("trj_report.html", "w") as handle:
        handle.write(report)


def run_orca_pltvib(fn, vib_indices):
    str_indices = [str(vi) for vi in vib_indices]
    cmd = "orca_pltvib {} {}".format(fn, " ".join(str_indices)).split()
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    stdout = result.stdout.decode("utf-8")
    created_re = "creating: (.+)\n"
    created_fns = re.findall(created_re, stdout)
    return created_fns


def imgvibs_from_orca_log(orca_log_fn):
    """
    Assuminig a non-linear molecule where the first imaginary frequency
    appears at index 6!.
    """
    orca_parser = Orca(orca_log_fn)
    index_range = range(6, 6+len(orca_parser.imgvibfreqs))
    trj_fns = run_orca_pltvib(orca_parser.fn, index_range)

    movie_fns = [movie_from_trajectory(trj_fn, "imgvib", trj_ind)
                 for trj_fn, trj_ind in zip(trj_fns, index_range)]

    imgvibs = [ImgVib(fn=orca_parser.fn, 
                      index=index,
                      value=float(value),
                      movie=movie)
                for index, value, movie
                in zip(index_range, orca_parser.imgvibfreqs, movie_fns)
    ]
    return imgvibs


def imgvibs_from_orca_hess(hess_fn):
    # Determine imganiary frequencies from the $ir_spectrum block
    # in the .hess file. The $ at the end matches either $end in
    # standalone frequency calculations or $job_list in optimization
    # runs.
    freq_re = "\$ir_spectrum\s*(\d+)\s*(.+?)\s*\$"
    with open(hess_fn) as handle:
        hess = handle.read()
    mobj = re.search(freq_re, hess, re.DOTALL)
    number_of_modes, ir_spectrum_str = mobj.groups()
    ir_spectrum_lines = ir_spectrum_str.strip().split("\n")
    allvibs = [float(line.strip().split()[0]) for line in ir_spectrum_lines]
    imgvib_indices = [i for i, iv in enumerate(allvibs) if iv < 0]
    imgvibs = [allvibs[i] for i in imgvib_indices]
    trj_fns = run_orca_pltvib(hess_fn, imgvib_indices)

    trj_dir_suf = "_" + os.path.basename(hess_fn.replace(".", "-"))

    movie_fns = [movie_from_trajectory(trj_fn, "imgvib", trj_ind, trj_dir_suf)
                 for trj_fn, trj_ind in zip(trj_fns, imgvib_indices)]

    imgvibs = [ImgVib(fn=hess_fn,
                      index=index,
                      value=float(value),
                      movie=movie_fn)
                for index, value, movie_fn
                in zip(imgvib_indices, imgvibs, movie_fns)
    ]
    return imgvibs


def save_imgvibs(imgvib_dict):
    with open("imgvibs.yaml", "w") as handle:
        handle.write(yaml.dump(imgvib_dict))

    create_imgvib_report(imgvib_dict)


def run():
    args = parse_args(sys.argv[1:])

    if args.imgvib:
        log_paths = search_files_with_ext(args.root_dir, ext=".out",
                                          ignore_fns=("slurm", ))
        imgvib_dict = {log_fn: imgvibs_from_orca_log(log_fn)
                       for log_fn in log_paths}
        save_imgvibs(imgvib_dict)
    if args.trj:
        trj_fns = natsorted(search_files_with_ext(args.root_dir, ext=".trj"))
        movie_fns = [movie_from_trajectory(trj_fn, "opt")
                     for trj_fn in trj_fns]
        create_trj_report(trj_fns, movie_fns)
    if args.hess:
        hess_fns = natsorted(search_files_with_ext(args.root_dir, ext=".hess"))
        imgvib_dict = {hess_fn: imgvibs_from_orca_hess(hess_fn)
                       for hess_fn in hess_fns}
        save_imgvibs(imgvib_dict)


if __name__ == "__main__":
    run()
