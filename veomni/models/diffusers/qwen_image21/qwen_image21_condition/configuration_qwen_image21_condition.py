from veomni.models.configuration_base import VeOmniPretrainedConfig


class QwenImage21ConditionModelConfig(VeOmniPretrainedConfig):
    model_type = "QwenImage21ConditionModel"

    def __init__(
        self,
        base_model_path: str = "",
        processor_subfolder: str = "processor",
        text_encoder_subfolder: str = "text_encoder",
        vae_subfolder: str = "vae",
        height: int | None = None,
        width: int | None = None,
        max_pixels: int = 1_048_576,
        num_train_timesteps: int = 1000,
        enable_edit: bool = False,
        seed: int = 42,
        text_encoder_attn_implementation: str = "sdpa",
        text_encoder_dtype: str = "bfloat16",
        vae_dtype: str = "bfloat16",
        max_sequence_length: int | None = None,
        **kwargs,
    ):
        if (height is None) != (width is None):
            raise ValueError("Set both Qwen-Image 2.1 height and width, or leave both unset.")
        if height is not None and (height <= 0 or width <= 0 or height % 32 or width % 32):
            raise ValueError("Qwen-Image 2.1 height and width must be positive multiples of 32.")
        if max_pixels < 32 * 32:
            raise ValueError("Qwen-Image 2.1 max_pixels must be at least 1024.")
        if num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least 2.")
        if kwargs.pop("image_resize_mode", "center_crop") != "center_crop":
            raise ValueError("Qwen-Image 2.1 uses DiffSynth center crop. Remove image_resize_mode from the config.")
        if max_sequence_length is not None and max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive or None.")
        for name, dtype in (("text_encoder_dtype", text_encoder_dtype), ("vae_dtype", vae_dtype)):
            if dtype not in ("float32", "bfloat16", "float16"):
                raise ValueError(f"{name} must be float32, bfloat16 or float16.")
        # Accept earlier DiffSynth configs without retaining the removed selector.
        if kwargs.pop("training_recipe", "diffsynth") != "diffsynth":
            raise ValueError("Qwen-Image 2.1 uses the DiffSynth schedule. Remove training_recipe from the config.")
        kwargs.pop("scheduler_subfolder", None)
        self.base_model_path = base_model_path
        self.processor_subfolder = processor_subfolder
        self.text_encoder_subfolder = text_encoder_subfolder
        self.vae_subfolder = vae_subfolder
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.num_train_timesteps = num_train_timesteps
        self.enable_edit = enable_edit
        self.seed = seed
        self.text_encoder_attn_implementation = text_encoder_attn_implementation
        self.text_encoder_dtype = text_encoder_dtype
        self.vae_dtype = vae_dtype
        self.max_sequence_length = max_sequence_length
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = dict(config_dict)
        for key in ("training_recipe", "scheduler_subfolder", "image_resize_mode", "height", "width", "max_pixels"):
            if key in kwargs:
                config_dict[key] = kwargs.pop(key)
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        # A Diffusers repository root has model_index.json instead of config.json.
        try:
            config, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        except OSError:
            config = {}
        config["base_model_path"] = str(pretrained_model_name_or_path)
        return config, kwargs
