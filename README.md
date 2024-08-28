<div align="center">
  <h1>Agent Task Memory 🏧</h1>
  <a href="https://img.shields.io/badge/arXiv-240x.xxxx-b31b1b.svg">
    <img src="https://img.shields.io/badge/arXiv-240x.xxxx-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://img.shields.io/badge/PRs-Welcome-red">
    <img src="https://img.shields.io/badge/PRs-Welcome-yellow" alt="PRs Welcome">
  </a>
</div>

Code for ["Agent Task Memory via Inducing and Wielding Workflows"](paper-arxiv-link).

<p align="center">
  <a href="https://zorazrw/agent-task-memory/">
    <img src="assets/teaser.jpg" width="90%" />
  </a>
</p>

## 🧠 What is Agent Task Memory?
Agent Task Memory (ATM) proposed to induce, integrate, and utilize workflows to the agent memory.
A workflow is usually a common sub-routine in solving tasks, with example-specific contexts being abstracted out.

ATM can operate in both offline and online settings:
- *offline* (left): when additional (e.g., training) examples are available, agents induces workflows from ground-truth annotated examples
- *online* (right): without any auxiliary data, agents induce workflows from past experiences on-the-fly.

<p align="center">
  <a href="https://zorazrw/agent-task-memory/">
    <img src="assets/online-offline.jpg" width="100%" />
  </a>
</p>

## 📈 How does ATM work?

### On WebArena
We achieve the state-of-the-art result -- 35.6% success rate.

<p align="center">
  <a href="https://zorazrw/agent-task-memory/">
    <img src="assets/webarena-bars.jpg" width="100%" />
  </a>
</p>

Check the code in `./webarena/` directory.

### On Mind2Web

We also get the best scores among text-based agents. Particularly, ATM offline effectively generalize across a wide range of tasks, websites, and domains.

<p align="center">
  <a href="https://zorazrw/agent-task-memory/">
    <img src="assets/mind2web-results.jpg" width="100%" />
  </a>
</p>

Check the code in `./mind2web/` directory.

## 📜 Citation

```bibtex
@inproceedings{agent-task-memory,
  title = {Agent Task Memory via Inducing and Wielding Workflows},
  author = {Wang, Zhiruo anf Mao, Jiayuan, and Fried, Daniel and Neubig, Graham},
  booktitle = {TBA},
  year = {2024},
  url = {TBA},
}
```
