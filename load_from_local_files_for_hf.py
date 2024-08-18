from transformers import AutoConfig, AutoModelForImageClassification
from pprint import pprint

if __name__ == "__main__":

    resnet50d_config = AutoConfig.from_pretrained("files_to_be_uploaded_to_hf/", trust_remote_code=True)
    pprint(resnet50d_config)

    resnet50d = AutoModelForImageClassification.from_pretrained("files_to_be_uploaded_to_hf/", trust_remote_code=True)
    pprint(resnet50d)