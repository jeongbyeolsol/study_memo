# huggingface

## class 정리

*Each pretrained model inherits from three base classes*

### [`PretrainedConfig`](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/configuration#transformers.PretrainedConfig)

모델이 어떤 하이퍼파라미터를 가지고 만들어졌는지를 정의하는 객체

### [`PreTrainedModel`](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel)

모든 사전학습(pretrained) 모델이 상속받는 추상 기반 클래스


### `Preprocessor`

모델에 문자열, 이미지, 오디오 등을 넣기 전에 모델이 이해할 형태로 바꾸는 단계

---
