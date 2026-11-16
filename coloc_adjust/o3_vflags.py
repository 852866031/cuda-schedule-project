#!/usr/bin/env python
"""Print the ft_worker.py flags of tenant variant V1 or V2 chosen by R2 (results_o3/r2_variants.json), or of a named variant.
Prints NONE if the slot is empty."""
import sys
from o3_common import VARIANTS, chosen_variants
k = sys.argv[1]; v = chosen_variants().get(k) if k in ("V1", "V2") else k
print("NONE" if not v else " ".join(["--bpt-variant", "base"] + VARIANTS[v]))
