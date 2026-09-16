<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 13_references.md
     PURPOSE: Bibliography
     DEPENDENCIES: All sections
     ═══════════════════════════════════════════════════════════════════════════ -->

# 13. References

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and 
          Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and 
          Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and memory-efficient exact attention 
         with IO-awareness},
  author={Dao, Tri and Fu, Daniel Y and Ermon, Stefano and Rudra, Atri and 
          R{\'e}, Christopher},
  booktitle={NeurIPS},
  year={2022}
}

@article{dao2023flashattention2,
  title={FlashAttention-2: Faster attention with better parallelism 
         and work partitioning},
  author={Dao, Tri},
  journal={arXiv preprint arXiv:2307.08691},
  year={2023}
}

@article{shah2024flashattention3,
  title={FlashAttention-3: Fast and accurate attention with asynchrony 
         and low-precision},
  author={Shah, Jay and Bikshandi, Ganesh and Zhang, Ying and 
          Thakkar, Vijay and Ramani, Pradeep and Dao, Tri},
  journal={arXiv preprint arXiv:2407.08608},
  year={2024}
}

@article{micikevicius2022fp8,
  title={FP8 formats for deep learning},
  author={Micikevicius, Paulius and Stosic, Dusan and Burgess, Neil and 
          Cornea, Marius and Dubey, Pradeep and Grisenthwaite, Richard and 
          Ha, Sangwon and Heinecke, Alexander and Judd, Patrick and 
          Kamalu, John and others},
  journal={arXiv preprint arXiv:2209.05433},
  year={2022}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and 
          Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and 
          Wang, Lu and Chen, Weizhu},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{dettmers2022llm,
  title={{LLM.int8()}: 8-bit matrix multiplication for transformers at scale},
  author={Dettmers, Tim and Lewis, Mike and Belkada, Younes and 
          Zettlemoyer, Luke},
  booktitle={NeurIPS},
  year={2022}
}

@article{frantar2023gptq,
  title={{GPTQ}: Accurate post-training quantization for generative 
         pre-trained transformers},
  author={Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and 
          Alistarh, Dan},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{lin2023awq,
  title={{AWQ}: Activation-aware weight quantization for {LLM}s},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and 
          Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and 
          Dang, Xingyu and Gan, Chuang and Han, Song},
  booktitle={MLSys},
  year={2024}
}

@inproceedings{kwon2023efficient,
  title={Efficient memory management for large language model serving 
         with {PagedAttention}},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and 
          Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E and 
          Zhang, Hao and Stoica, Ion},
  booktitle={SOSP},
  year={2023}
}

@inproceedings{rabe2021self,
  title={Self-attention does not need {O}(n²) memory},
  author={Rabe, Markus N and Staats, Charles},
  journal={arXiv preprint arXiv:2112.05682},
  year={2021}
}

@article{peng2023fp8lm,
  title={{FP8-LM}: Training {FP8} large language models with 
         automatic mixed precision},
  author={Peng, Houwen and Wu, Kan and Wei, Yixuan and Zhao, Guoshuai and 
          Tan, Yurong and Yang, Ming and Liu, Ruobing and Xiong, Yuxuan and 
          Sun, Mingjie and Li, Binhang and others},
  journal={arXiv preprint arXiv:2310.18313},
  year={2023}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a 
         sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and 
          Chuang, Jason and Manning, Christopher D and Ng, Andrew Y and 
          Potts, Christopher},
  booktitle={EMNLP},
  year={2013}
}

@article{sanh2019distilbert,
  title={{DistilBERT}, a distilled version of {BERT}: smaller, faster, 
         cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and 
          Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title={{RoBERTa}: A robustly optimized {BERT} pretraining approach},
  author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and 
          Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis, Mike and 
          Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and 
          Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Blog},
  year={2019}
}

@article{su2021roformer,
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Murtadha, Ahmed and 
          Wen, Bo and Liu, Yunfeng},
  journal={arXiv preprint arXiv:2104.09864},
  year={2021}
}

@article{shazeer2019fast,
  title={Fast transformer decoding: One write-head is all you need},
  author={Shazeer, Noam},
  journal={arXiv preprint arXiv:1911.02150},
  year={2019}
}
```

---