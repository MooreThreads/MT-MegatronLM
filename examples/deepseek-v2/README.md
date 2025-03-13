# Deeoseek-v2
We implement all kinds of aux loss mentioned in deepseek-v2 paper, which contains:
- **seq-aux-loss**: used with args "--moe-router-load-balancing-type seq_aux_loss"
- **moe-device-level-aux-loss-coeff**
- **moe-comm-aux-loss-coeff**

In **deepseekv2-lite**, we follow the setting in HuggingFace which only use pipeline parallel, and "device-level-loss", "comm-aux-loss" will not be used.

### Innovation
In addition to the original implementation, we added a device-level drop strategy to replace the expert-level drop strategy, which can avoid discarding too many tokens while maintaining balance, although this will apparently reduce the MFU.