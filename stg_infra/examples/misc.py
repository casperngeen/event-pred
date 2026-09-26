import sys
sys.path.insert(0, "stg_infra/examples")
from pairwise_monotonicity_taker_side_check_v2 import classify_subtitle
print(classify_subtitle("Category 3 or above"))  # expect: upper_tail
print(classify_subtitle("Chicago: 44 or above and New York: 71 or above"))  # expect: unrecognized