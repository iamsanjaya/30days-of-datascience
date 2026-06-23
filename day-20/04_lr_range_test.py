# %%
"""Day 20 — 04 (out-of-box): LR Range Test (Leslie Smith, 2017).

Train for one epoch while exponentially ramping the learning rate from
config.LR_RANGE_START to config.LR_RANGE_END. Plot loss vs LR to read off:
  1. the LR where loss starts decreasing
  2. the steepest-descent LR — the best candidate to actually train with
  3. the LR where loss diverges

Compare point (2) against the LR you tuned manually in Day 19's architecture
search and note the match (or mismatch) in your learning journal.
"""

import sys
from math import ceil
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import config
from utils import architecture, data, training, visualization

# %% Full train pool — the range test needs many batches in one epoch, not the
# tiny 500-image overfit subset.
(x_train, y_train), (x_val, y_val) = data.get_full_train_val()
x_train = data.normalize(x_train)
print(f"Train pool for LR sweep: {x_train.shape}")

# %% Set up the exponential LR ramp across exactly one epoch
# ceil, not floor: Keras still runs a partial last batch (e.g. 40000/128 =
# 312.5 -> 313 actual steps), so floor division would compute the per-step
# multiplier for one fewer step than actually happens.
total_batches = ceil(len(x_train) / config.LR_RANGE_BATCH_SIZE)
lr_callback = training.LRRangeTestCallback(
    start_lr=config.LR_RANGE_START,
    end_lr=config.LR_RANGE_END,
    total_iterations=total_batches,
)

# Use the best-performing architecture from 03 (BatchNorm) as the test model.
model = architecture.build_regularized_cnn(batchnorm=True)
model = training.compile_model(model, learning_rate=config.LR_RANGE_START)

# %% Train exactly one epoch — the callback handles the LR ramp per batch
print("Note: the aggregate loss/accuracy Keras prints below averages across")
print("the whole epoch, during which LR varied by 8 orders of magnitude —")
print("it doesn't describe model quality. The per-batch lr/loss curve below does.")

model.fit(
    x_train,
    y_train,
    batch_size=config.LR_RANGE_BATCH_SIZE,
    epochs=1,
    callbacks=[lr_callback],
    verbose=2,
)

# %% Identify landmark LRs and plot
landmarks = visualization.find_lr_range_test_points(lr_callback.lrs, lr_callback.losses)
for name, value in landmarks.items():
    print(f"{name}: {value:.2e}")

visualization.plot_lr_range_test(lr_callback.lrs, lr_callback.losses, landmarks)

print(
    "\nCompare 'lr_steepest_descent' above against the learning rate you tuned"
    " manually in Day 19 — record the comparison in your learning journal."
)
