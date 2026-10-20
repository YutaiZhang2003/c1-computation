"""Install the classified genus-one source directories."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/one_loop/three_point/other_scripts'))
from genus1_source_paths import install
install()
