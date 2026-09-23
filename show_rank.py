# show_rank.py
import os
print(os.environ.get('LOCAL_RANK'), os.environ.get('RANK'), os.environ.get('WORLD_SIZE'))
