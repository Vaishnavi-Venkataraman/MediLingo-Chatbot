---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:960
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: i am overweight and have noticed that my legs are swollen and the
    blood vessels are visible my legs have swollen and i can see a stream of swollen
    veins on my calves
  sentences:
  - the swelling in my legs is causing me to have difficulty fitting into my shoes
    i cant sprint or stand for long periods of time i can see some swollen blood vessels
  - there is a silver like dusting on my skin moreover the skin on my arms and back
    are starting to peel off this is strange and really concerning me
  - ive been having a lot of problems using the restroom recently its been excruciatingly
    uncomfortable and ive been feeling agony in my anus my stool has been bloody as
    well and my anus has been really inflamed
- source_sentence: along with recurrent headaches and blurred vision i suffer acid
    reflux and trouble digesting my food
  sentences:
  - along with excessive appetite a stiff neck depression impatience and visual disturbance
    ive also been suffering indigestion headaches blurred vision and stomach problems
  - i cant stop sneezing and im exhausted and sick my throat is really uncomfortable
    and there is a lot of junk in my nose and throat my neck is also swollen and puffy
  - i have been experiencing intense itching vomiting and fatigue i have also lost
    weight and have a high fever my skin has turned yellow and my urine is dark i
    have been having abdominal pain as well
- source_sentence: the high fever swollen lymph nodes and headache are causing me
    a lot of trouble i dont feel like eating anything and feel weak and fatigued its
    hard for me to concentrate on my daily life
  sentences:
  - i have been suffering from back pain a chronic cough and weakness in my arms and
    legs my neck hurts and i have been feeling dizzy and off balance
  - ive recently struggled with a really irritating skin rash there are blackheads
    and pusfilled pimples all over it additionally my skin has been scurring a lot
  - i have a skin rash thats red and inflamed and its spreading all over my body ive
    been experiencing intense itching especially on my arms and legs
- source_sentence: my nose always feels stuffy and congested and my eyes are always
    red and itching i have a feeling of being unwell and fatigued and i keep hacking
    up this gunk i have a scratchy irritated throat and ive seen that my necks bumps
    are larger than usual
  sentences:
  - my eyes are usually red and runny and my nose is always stuffy ive also been having
    difficulty breathing and my chest hurts in addition i cant smell anything and
    my muscles are quite painful
  - ive got a cough that wont go away and im exhausted ive been coughing up thick
    mucous and my fever is also pretty high
  - i experience skin irritations and rashes especially in my skins creases any wounds
    and bruises i have on my skin also heal quite slowly
- source_sentence: ive been experiencing a lot of problems with my bowel motions recently
    its difficult to go and it hurts when i do my anus is quite painful and it has
    been bleeding whenever i go its excruciatingly painful and im quite uneasy
  sentences:
  - i have been vomiting frequently and have lost my appetite as a result there are
    rashes on my skin and my eyes pain because of which i cannot sleep properly
  - i have asthma like symptoms like wheezing and difficulty breathing i often get
    fever and have headaches i feel tired all the time
  - my bowel motions are giving me a lot of problems right now going is difficult
    and going hurts when i do it when i go my anus bleeds and is really uncomfortable
    im in a lot of discomfort and it hurts extremely bad
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'ive been experiencing a lot of problems with my bowel motions recently its difficult to go and it hurts when i do my anus is quite painful and it has been bleeding whenever i go its excruciatingly painful and im quite uneasy',
    'my bowel motions are giving me a lot of problems right now going is difficult and going hurts when i do it when i go my anus bleeds and is really uncomfortable im in a lot of discomfort and it hurts extremely bad',
    'i have asthma like symptoms like wheezing and difficulty breathing i often get fever and have headaches i feel tired all the time',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9616, 0.4749],
#         [0.9616, 1.0000, 0.4977],
#         [0.4749, 0.4977, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 960 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 960 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 14 tokens</li><li>mean: 36.38 tokens</li><li>max: 63 tokens</li></ul> | <ul><li>min: 14 tokens</li><li>mean: 36.38 tokens</li><li>max: 63 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                 | sentence_1                                                                                                                                                                                                                                                            |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>recently ive felt quite weak and exhausted and ive also had a cough that simply wont go away my fever has been really high and it has been challenging for me to catch my breath ive been making a lot of mucous when i cough</code> | <code>ive felt terribly weak and drained and ive also had a cough that that wont go away my fever has been exceptionally high and it has been challenging to try to catch my breath ive been creating a lot of mucous when i cough</code>                             |
  | <code>ive been quite exhausted and ill my throat has been quite painful and ive had a fairly nasty cough ive got a lot of chills and a pretty high temperature just feeling extremely run down and weak</code>                             | <code>my eyes are red and watery all the time ive also had this pressure in my sinuses that wont go away im always feeling tired and ive been having a lot of trouble breathing ive also had a lot of gunk in my throat and my lymph nodes are swollen</code>         |
  | <code>ive been coughing a lot and finding it difficult to breathe my throat hurts and i feel like i have a lot of phlegm trapped in my chest my nose has been running a lot and ive been feeling really congested</code>                   | <code>my nose always feels stuffy and congested and my eyes are always red and itching i have a feeling of being unwell and fatigued and i keep hacking up this gunk i have a scratchy irritated throat and ive seen that my necks bumps are larger than usual</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.13.3
- Sentence Transformers: 5.1.1
- Transformers: 4.56.2
- PyTorch: 2.8.0+cpu
- Accelerate: 
- Datasets: 
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->