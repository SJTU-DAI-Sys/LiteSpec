"""Train DLite sft."""

from litespec.training import run_training

if __name__ == "__main__":
    run_training("dlite", mode="sft")
