"""`python -m janus …` 与 `janus …` 等价，见 janus/cli.py。"""

import sys

from janus.cli import main

sys.exit(main())
