# 模型角色审计 #04 — openai-community/gpt2

基本信息
- 审计基线: vLLM main @ c7ce03bcbd
- 模型实现: `vllm/model_executor/models/gpt2.py`(381 行)
- 架构要点: decoder-only,**learned positional embeddings(无 RoPE)**,HF 侧用 Conv1D(权重转置存储),tied lm_head
- 官方审计草稿: 无(本模型没有官方 draft 可对照,本文为独立审计)

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `wte`(VocabParallelEmbedding) | gpt2.py:197-202 | R1 | P3 copy-back | RESTORABLE | 无 |
| `wpe`(nn.Embedding,learned pos emb) | gpt2.py:203 | **R1 checkpoint 参数**(HF key `wpe.weight`,非派生 buffer) | P3 copy-back | RESTORABLE | 无 |
| `c_attn` / `c_proj` / `c_fc` 权重与 bias | gpt2.py:78-100, 123-136 | R1,但**加载时需转置**(见特殊发现 1) | P3 copy-back(须为转置后形态) | RESTORABLE(带 load-transform 约束) | 低(仅当 reload 绕过 `load_weights` 时出错) |
| `ln_1` / `ln_2` / `ln_f` | gpt2.py:158, 162, 209 | R1 | P3 copy-back | RESTORABLE | 无 |
| `lm_head`(tie_word_embeddings 时 `tie_weights(wte)`) | gpt2.py:273-280 | R1(别名) | 参数对象共享 | RESTORABLE(reload 须原位写) | 无 |
| `score`(GPT2ForSequenceClassification,nn.Linear) | gpt2.py:335-340 | R1 | P3 copy-back | RESTORABLE | 无 |
| HF checkpoint 中的 `.attn.bias` / `.attn.masked_bias` 因果掩码 | gpt2.py:255-259(skip_substrs) | 不落地:vLLM 模型未注册这些 buffer,加载时直接跳过 | N/A | N/A | 无 |
| Attention `_q/_k/_v/_prob_scale` buffers(公共层) | attention/attention.py:127-130, 184 | R2 | P2 named_buffers 备份/回填(gpu_worker.py:270-275, 311-316) | RESTORABLE | 无 |

## 特殊发现

1. **Conv1D 转置是 load-path 转换,不是驻留状态,但对 reload 机制有硬约束**:`GPT2Model._transpose_conv1d`(gpt2.py:242-253)在权重流入时对 `c_attn`/`c_proj`/`c_fc` 的 2D 权重做 `loaded_weight.t()`。这意味着:
   - GPU 上的参数形态与 checkpoint 磁盘形态**不同构**(互为转置);
   - reload 若走 `model.load_weights()`(P4 路径)则转置被重放,正确;
   - reload 若实现为"用 checkpoint 原始张量直接 `copy_` 到 named_parameters"(绕过 load_weights),会得到转置错误的权重且**形状恰好合法**(方阵 c_proj)或在非方阵处报 shape mismatch——前者是静默数值错误。权重生命周期重设计需把"load-transform 存在与否"作为模型级元信息。
   - 同文件注释(gpt2.py:247)已自认 "the logic below might break quantized models"。
2. 两个 load_weights 入口(`GPT2LMHeadModel.load_weights` gpt2.py:309-312 与 `GPT2Model.load_weights` gpt2.py:255-260)只有内层做转置;顶层通过 AutoWeightsLoader 递归会命中内层,路径一致。`_add_transformer_prefix`(gpt2.py:371-381)只是命名重写,无状态。
3. 与 OPT 相同,tied lm_head(gpt2.py:279-280 `tie_weights`)依赖参数对象同一性,reload 不得替换参数对象。
4. 无 register_buffer、无 RoPE、无 PWAL 自定义、无 kernel 常量;`get_act_fn`(gpt2.py:137)无状态。

## 结论

GPT-2 对 sleep level-2 本身无风险状态(与 OPT 同为近似 null case)。唯一值得在重构中显式建模的是 **Conv1D load-time 转置**:它证明 R1 不总等价于 "checkpoint 字节可直拷",P3 copy-back 的备份源必须取自 GPU 上已转换的参数(或强制经 load_weights 重放)。建议在新框架中将 GPT-2 作为 "R1 + load-transform" 的最小测试用例。
