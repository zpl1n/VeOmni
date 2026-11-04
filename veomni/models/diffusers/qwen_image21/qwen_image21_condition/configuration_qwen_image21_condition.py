from veomni.models.configuration_base import VeOmniPretrainedConfig


class QwenImage21ConditionModelConfig(VeOmniPretrainedConfig):
    model_type = "qwen_image21"

    def __init__(
        self,
        base_model_path: str = "",
        processor_subfolder: str = "processor",
        text_encoder_subfolder: str = "text_encoder",
        vae_subfolder: str = "vae",
        scheduler_subfolder: str = "scheduler",
        height: int = 1024,
        width: int = 1024,
        num_train_timesteps: int = 1000,
        training_recipe: str = "diffusers",
        enable_edit: bool = False,
        seed: int = 42,
        image_resize_mode: str = "resize",
        text_encoder_attn_implementation: str = "eager",
        text_encoder_dtype: str = "bfloat16",
        vae_dtype: str = "float32",
        max_sequence_length: int | None = None,
        **kwargs,
    ):
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError("Qwen-Image 2.1 height and width must be positive multiples of 32.")
        if num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least 2.")
        if image_resize_mode not in ("resize", "center_crop"):
            raise ValueError("image_resize_mode must be 'resize' or 'center_crop'.")
        if max_sequence_length is not None and max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive or None.")
        for name, dtype in (("text_encoder_dtype", text_encoder_dtype), ("vae_dtype", vae_dtype)):
            if dtype not in ("float32", "bfloat16", "float16"):
                raise ValueError(f"{name} must be float32, bfloat16 or float16.")
        if training_recipe not in ("diffusers", "diffsynth"):
            raise ValueError("training_recipe must be 'diffusers' or 'diffsynth'.")
        self.training_recipe = training_recipe
        self.base_model_path = base_model_path
        self.processor_subfolder = processor_subfolder
        self.text_encoder_subfolder = text_encoder_subfolder
        self.vae_subfolder = vae_subfolder
        self.scheduler_subfolder = scheduler_subfolder
        self.height = height
        self.width = width
        self.num_train_timesteps = num_train_timesteps
        self.enable_edit = enable_edit
        self.seed = seed
        self.image_resize_mode = image_resize_mode
        self.text_encoder_attn_implementation = text_encoder_attn_implementation
        self.text_encoder_dtype = text_encoder_dtype
        self.vae_dtype = vae_dtype
        self.max_sequence_length = max_sequence_length
        super().__init__(**kwargs)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        # A Diffusers repository root has model_index.json instead of config.json.
        try:
            config, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        except OSError:
            config = {}
        config["base_model_path"] = str(pretrained_model_name_or_path)
        return config, kwargs
