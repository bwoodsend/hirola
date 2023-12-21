import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__, "..", "..").resolve()))

import numpy as np
from sloth.timers import Stopwatch
from peccary import Scene

from tests import generate
from tests.benchmarks import METHODS

parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=argparse.FileType("w"))
parser.add_argument("generator",
                    choices=[i.__name__ for i in generate.generators])
options = parser.parse_args()


def time(f, n, *data):
    watch = Stopwatch()
    watch.start()
    out = [0]
    for i in range(n):
        f(*data)
        out.append(watch.lap())
    return np.diff(out)[-5:]


def round_(x):
    return [np.round(x, 4 - int(np.log10(x))) for x in x]


generator = getattr(generate, options.generator)

sizes = np.unique(np.logspace(0, 7, 50).astype(int))
times = {method: {} for method in METHODS}
too_slow = set()

for size in sizes:
    print(size, file=sys.stderr)
    try:
        data = generator(int(size))
    except generate.DataLimitExeeded:
        break
    iterations = int(max(10 // size, 1))
    for (name, (method, normalize)) in METHODS.items():
        if name in too_slow:
            continue
        _data = normalize(data) if normalize else data
        times[name][size] = time(method, iterations, _data)
        if min(times[name][size]) > 1:
            too_slow.add(name)

scene = Scene(
    xaxis={
        "title": {
            "text": "Number of keys"
        },
        "type": "log"
    },
    yaxis={
        "title": {
            "text": "Time per key (seconds)"
        },
        "type": "log"
    },
    height=600,
)

for method in times:
    x = []
    y = []
    for (size, _times) in times[method].items():
        x.append(np.full(len(_times), size))
        y.append(_times / size)
    scene.plot(x=np.concatenate(x), y=round_(np.concatenate(y)), name=method,
               type="scatter", mode="markers+lines", marker={"size": 4},
               line={"shape": "spline"})

if options.output:
    with options.output as f:
        f.write(scene.to_html())
else:
    scene.preview()
