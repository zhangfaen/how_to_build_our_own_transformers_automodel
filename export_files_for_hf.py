from faen_resnet_model.configuration_resnet import ResnetConfig
from faen_resnet_model.modeling_resnet import ResnetModelForImageClassification, ResnetModel
import timm

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

if __name__ == "__main__":
    # Test
    # AutoConfig.register("faen_resnet", ResnetConfig)
    # AutoModel.register(ResnetConfig, ResnetModel)
    # AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)

    ResnetConfig.register_for_auto_class()
    ResnetModel.register_for_auto_class("AutoModel")
    ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")

    resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
    resnet50d = ResnetModelForImageClassification(resnet50d_config)

    pretrained_model = timm.create_model("resnet50d", pretrained=True)
    resnet50d.model.load_state_dict(pretrained_model.state_dict())

    resnet50d.save_pretrained("files_to_be_uploaded_to_hf/")
    resnet50d_config.save_pretrained("files_to_be_uploaded_to_hf/")