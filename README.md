# How to build a our own pretraind model with huggingface transformers lib and load it by AutoModel.from_pretrained

In this repo, we will show how to write a custom model and its configuration so it can be used inside Transformers, and how we can share it with the community (with the code it relies on) so that anyone can use it, even if it’s not present in the 🤗 Transformers library. We’ll see how to build upon transformers and extend the framework with withß hooks and custom code.

We will illustrate all of this on a ResNet model, by wrapping the ResNet class of the timm library into a PreTrainedModel.

Talk is cheap, let's see code!

```bash
%python export_files_for_hf.py
%python load_from_local_files_for_hf.py
%huggingface-cli upload zhangfaen/faen_resnet_model files_to_be_uploaded_to_hf/ 
%python load_from_files_at_hf.py
```