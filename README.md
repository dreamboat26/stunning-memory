# Instruct-Tune TinyLlama

<p align="center">
  <img src="./assets/sample.png" height="70%" width="70%"/>
</p>

This repository contains an easy-to-use and understand code to instruct-tune, or SFT (supervised fine tuning) [TinyLlama](https://github.com/jzhang38/TinyLlama) to create ChatGPT like chatbots (stateless).

I'll be using TinyLlama, however as stated in their repository it uses the same architecture as [Llama 2](https://arxiv.org/abs/2307.09288). If you want to use Llama 2 it'll be probably fine.

We'll be using both [Alpaca Dataset](https://crfm.stanford.edu/2023/03/13/alpaca.html), and a custom dataset that we'll load from our local machine.

Prompt format we'll be using is
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Input:
{context}

### Response:
```
