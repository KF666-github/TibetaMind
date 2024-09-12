# TibetaMind
**TibetaMind** is an advanced language model based on the Llama 3-8B-Instruct architecture, further fine-tuned using extensive Tibetan language corpora. Through this specialized fine-tuning, **TibetaMind** has significantly enhanced its ability to comprehend, process, and generate Tibetan language content, while also providing seamless cross-language understanding between Tibetan and Chinese. This allows for accurate translation and communication across these languages. **TibetaMind** can be applied to a variety of tasks, including Tibetan text generation, summarization, and translation between Tibetan and Chinese, playing a pivotal role in preserving and advancing Tibetan linguistics in the digital age.

# How to use
## Use with transformers
### Transformers AutoModelForCausalLM
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "DaydreamerF/TibetaMind"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "如何用藏语表达下面汉语的意思：汉语句子：大狗在楼里不好养。"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```
