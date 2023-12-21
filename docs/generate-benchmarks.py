from pathlib import Path
import textwrap
import subprocess
import sys

with open(Path(__file__, "../source/_benchmarks.txt").resolve(), "w") as f:
    for (generator, description) in [
        ("random_16", "Highest entropy short strings of 16 random bytes."),
        ("random_128", "Highest entropy long strings of 128 random bytes."),
        ("id_like", "Low entropy pairs of small 32 bit integers."),
        ("permutative",
         "Lowest entropy all possible pairs of integers from ``(0, 0)`` to ``(sqrt(n), sqrt(n))``."
        ),
        ("floating_32",
         "High entropy 32-bit triplets of floats from -30 to 30."),
        ("floating_64", "Like ``floating_32`` but with 64-bit floats."),
        ("textual",
         "Short (up to 20 character) non-random strings from a `common passwords database <https://github.com/kkrypt0nn/wordlists/raw/1ca65fea80381e2caf9031e02c0602da6b48e936/wordlists/passwords/bt4_passwords.txt>`_."
        ),
        ("utf8", "Like ``textual`` but using bytes instead of strings."),
    ]:
        p = subprocess.run([
            sys.executable,
            Path(__file__, "../../tests/benchmark_graphs.py").resolve(), "-o-",
            generator
        ], stdout=subprocess.PIPE, check=True)
        f.write(
            textwrap.dedent(f"""
            {generator}
            ---------------

            {description}

            .. raw:: html

                <style>
                    .wy-nav-content {{
                        max-width: unset;
                    }}
                </style>
        """))
        f.write(textwrap.indent(p.stdout.decode(), "    "))
