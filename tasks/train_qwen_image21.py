"""Train Qwen-Image 2.1 with model-specific MFU logging."""

from veomni.arguments import parse_args
from veomni.models.diffusers.qwen_image21.mfu import QwenImage21MFUArguments, QwenImage21MFUTrainer


if __name__ == "__main__":
    args = parse_args(QwenImage21MFUArguments)
    trainer = QwenImage21MFUTrainer(args)
    trainer.train()
