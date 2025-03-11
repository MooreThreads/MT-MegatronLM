# MT-MegatronLM


## Installation
You can create a directory named `megatron_dev,` and use the command below to clone the `Megatron-LM`, `MT-Megatron`LM to `megatron_dev`.  

```bash
# Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
pushd Megatron-LM
git checkout -b core_r0.9.0 core_r0.9.0
popd

# megatron-lm-musa-patch
git clone https://github.com/MooreThreads/MT-MegatronLM.git
pushd MT-MegatronLM
popd

## Getting started
### Llama3 

```bash
cd MT-MegatronLM/examples/llama3
bash dist_run_pretrain_megatron_llama3_musa.sh
```

### Mixtral

```bash
cd MT-MegatronLM/examples/mixtral
bash dist_run_pretrain_megatron_llama3_musa.sh
```

### Llava

```bash
cd MT-MegatronLM/examples/llava

```

### DeepSeekV3

```bash
cd MT-MegatronLM/examples/deepseekv3

```
In deepseek-v2/v3, the ffn-size in first several dense layer is not the same as moe-ffn-size. So it's need to modify some codes in Megatron to support this situation while not use GroupGEMM.
#### Modify some codes in Megatron

Megatron-LM/megatron/core/transformer/mlp.py

add in line63:  
```
if is_expert:
    ffn_hidden_size = self.config.moe_ffn_hidden_size
```
change in line83:
```
            self.config.ffn_hidden_size,
-->         self.config.ffn_hidden_size if not is_expert else self.config.moe_ffn_hidden_size,
```


Megatron-LM/megatron/core/transformer/moe/experts.py

comment line757-760
```
        # assert (
        #     self.config.moe_ffn_hidden_size == self.config.ffn_hidden_size
        # ), "Please use GroupedMLP or TEGroupedMLP when moe_ffn_hidden_size is \
        #         different from ffn_hidden_size"
```
