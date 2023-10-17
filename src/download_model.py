import os

from dataclasses import dataclass, field
import transformers as tfs


@dataclass
class Arguments:
    model_name: str = field(
        metadata={"help": "Model name on HuggingFace. Eg. `t5-base`."}
    )
    output_dir: str = field(
        metadata={"help": "Output parent directory for the model and tokenizer."}
    )

parser = tfs.HfArgumentParser((Arguments))
args = parser.parse_args_into_dataclasses()[0]

output_dir = os.path.join(args.output_dir, args.model_name)
if os.path.exists(output_dir):
    print(f'{output_dir} exists!')
    exit()

print('='*100)
print(' '*10, f'Download {args.model_name} model and tokenizer')
print('='*100)
#m = tfs.AutoModelForMaskedLM.from_pretrained(args.model_name)
m = tfs.AutoModel.from_pretrained(args.model_name)
t = tfs.AutoTokenizer.from_pretrained(args.model_name)

print('='*100)
print(' '*10, f'Save {args.model_name} model and tokenizer')
print('='*100)
m.save_pretrained(output_dir)
t.save_pretrained(output_dir)