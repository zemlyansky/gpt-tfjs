import json
import os
import pickle

import torch
from torch.utils.data import Dataset

os.sys.path.append('./mingpt')
from mingpt.model import NewGELU as GELU, CausalSelfAttention, GPT
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN, set_seed
set_seed(3407)

# Create test decorator to parse input from string to tensor
# And also print the output
def test(func):
    def wrapper(x: str):
        x = json.loads(x)
        x = torch.tensor(x, dtype=torch.float32)
        print(func(x))
    return wrapper

def convert_config_to_js(config):
    mapper = {
        'bias': 'bias',
        'block_size': 'blockSize',
        'dropout': 'dropout',
        'n_embd': 'nEmbd',
        'n_head': 'nHead',
        'n_layer': 'nLayer',
        'vocab_size': 'vocabSize'
    }
    return {mapper[k]: v for k, v in config.items()}


def gen_gelu():
    print('Generating test: GELU')
    inputs = [
        -1,
        [1, 2, 3],
        [[-100, 0, 100], [1, 2, 3]],
        [[[-10000, 0, 10000], [1, 2, 3]], [[-10000, 0, 10000], [1, 2, 3]]],
    ]
    gelu = GELU()
    outputs = [gelu(torch.tensor(input, dtype=torch.float32)).tolist() for input in inputs]
    return {
        'inputs': inputs,
        'outputs': outputs,
    }

def gen_att():
    print('Generating test: CausalSelfAttention')
    configs = [
        {
            'block_size': 2,
            'n_embd': 8,
            'n_head': 4,
            'bias': True,
            'dropout': 0,
        },
        {
            'block_size': 4,
            'n_embd': 4,
            'n_head': 2,
            'dropout': 0,
            'bias': True,
        },
        {
            'block_size': 4,
            'n_embd': 4,
            'n_head': 2,
            'dropout': 0,
            'bias': True,
        },
    ]
    inputs = [
        [[[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]],
        [[[-100, -80, -60, -40], [-20, 0, 20, 40], [-100, 0, 20, 40], [-80, -60, 20, 40]]],
        [
            [[-100, -80, -60, -40], [-20, 0, 20, 40], [-100, 0, 20, 40], [-80, -60, 20, 40]],
            [[-69, 10, 20, 30], [40, 50, 60, 70], [80, 90, 100, 110], [120, 130, 140, 150]]
        ]
    ]
    outputs = []
    for input, config in zip(inputs, configs):
        config = CN(
            **config,
            attn_pdrop=configs[0]['dropout'], 
            resid_pdrop=configs[0]['dropout']
        )
        att = CausalSelfAttention(config)
        for p in att.parameters():
            p.data.fill_(1)
        outputs.append(att(torch.tensor(input, dtype=torch.float32)).tolist())
    return {
        'inputs': inputs,
        'outputs': outputs,
        'configs': [convert_config_to_js(config) for config in configs],
    }

def gen_att_grads():
    print('Generating test: CausalSelfAttention grads')
    configs = [{
        'block_size': 3,
        'n_embd': 4,
        'n_head': 2,
        'dropout': 0,
        'bias': True,
    }]
    inputs = [
        [[[-100, -80, -60, -40], [-20, 0, 20, 40], [-100, 0, 20, 40]]]
    ]
    config = CN(
        **configs[0], 
        attn_pdrop=configs[0]['dropout'], 
        resid_pdrop=configs[0]['dropout']
    )
    input = torch.tensor(inputs[0], dtype=torch.float32)
    att = CausalSelfAttention(config)
    for p in att.parameters():
        p.data.fill_(1)
    output = att(input)
    loss = torch.mean((output - input) ** 2)
    loss.backward()
    # Prepare grads and transpose dense to match TFJS
    # That's a crazy difference between PyTorch and TFJS
    # Dense layer in TFJS returns weights with shape [n_input, n_output]
    # Dense layer in PyTorch returns weights with shape [n_output, n_input]
    # PyTorch seems natural as n_output is the number of neurons
    grads = [
        p.grad.tolist() if len(p.grad.shape) == 1 else p.grad.transpose(0, 1).tolist() 
        for p in att.parameters()
    ]
    print('Grads:', grads)
    return {
        'inputs': inputs,
        'configs': [convert_config_to_js(configs[0])],
        'grads': [grads]
    }

class SortDataset(Dataset):
    """ 
    From: https://github.com/karpathy/minGPT/blob/master/demo.ipynb 
    """
    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.num_digits = num_digits
    
    def __len__(self):
        return 10000
    
    def get_vocab_size(self):
        return self.num_digits
    
    def get_block_size(self):
        return self.length * 2 - 1

    def __getitem__(self, idx):
        while True:
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    continue
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = 'test' if h % 4 == 0 else 'train'
            if inp_split == self.split:
                break # ok
        sol = torch.sort(inp)[0]
        cat = torch.cat((inp, sol), dim=0)
        x = cat[:-1].clone()
        y = cat[1:].clone()
        y[:self.length-1] = -1
        return x, y

def gen_model_sort():
    print('Generating test: Sort model (minGPT)')
    # Use mingpt example
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'

    train_dataset = SortDataset('train')
    test_dataset = SortDataset('test')

    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model = GPT(model_config)

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    train_config.max_iters = 1000
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, train_dataset)

    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run()

    # torch.save(model.state_dict(), './temp/model_sort_fin.pt')
    
    # Without this line, the model will not be in eval mode
    # And dropout will be applied
    # Source of some painful to debug differences in results
    model.eval()   
    
    state_dict = model.state_dict()
    weights_names = list(state_dict.keys())
    for wn in weights_names:
        print(wn, state_dict[wn].shape)

    n = train_dataset.length

    inputs = [[0, 0, 2, 1, 0, 1]] 
    inputsPT = torch.tensor(inputs, dtype=torch.long).to(trainer.device)

    with torch.no_grad():
        logits, _ = model(inputsPT)
        cat = model.generate(inputsPT, n, do_sample=False)
    sol = cat[:, n:]

    print(logits.tolist())

    return {
        'config': vars(model_config),
        'weights': {k: v.tolist() for k, v in state_dict.items()},
        'inputs': inputs,
        'outputs': sol.tolist(),
        'logits': logits.tolist()
    }

def main():
    tests = {
        'gelu': gen_gelu(),
        'att': gen_att(),
        'att_grads': gen_att_grads(),
        'model_sort': gen_model_sort(),
    }
    with open('test.json', 'w') as f:
        json.dump(tests, f)

if __name__ == '__main__':
    main()