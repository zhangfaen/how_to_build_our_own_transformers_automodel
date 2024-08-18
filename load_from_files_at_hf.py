from transformers import AutoConfig, AutoModel, AutoModelForImageClassification
from pprint import pprint

if __name__ == "__main__":
    resnet50d_config = AutoConfig.from_pretrained("zhangfaen/faen_resnet_model", trust_remote_code=True)
    pprint(resnet50d_config)

    resnet50d = AutoModelForImageClassification.from_pretrained("zhangfaen/faen_resnet_model", trust_remote_code=True)
    pprint(resnet50d)