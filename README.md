# Infinite-Latent-Feature-Models-and-the-Indian-Buffet-Process

Author: Linlin Li & Bo Liu

Project for STA663: Infinite Latent Feature Models and the Indian Buffet Process

This repository includes a module IBP which helps make inference on infinite latent feature models using the Indian Buffet Process. The algorithm was proposed by Thomas L. Griffiths and Zoubin Ghahramani. More information on this package can be referenced in `paper/FinalReport.pdf`.

## Test

Two test files on calculating log conditional probability and sample K_plus are provided in `test/test_lp.py` and `test/recursion.py`. Both python files include functions for test. These functions can be explicitly called individually, or the file can be directly run via terminal.

```console
foo@bar:IBP4ILFM/test foo$ python test_lp -N 100 -D 100 -K 10 -T 100 [--sigma_X=1 --sigma_A=1]
foo@bar:IBP4ILFM/test foo$ python test_recursion -N 100 -D 100 -K 10 -C 10 -T 100
```

- N, D: Dimension of input array;
- K: Number of latent features;
- T: Number of repetitions;
- C: Maximum number of K_plus;
- sigma_X, sigma_A: [optional, random value in (0,1) if undefined] Standard deviation for prior normal distribution.

## Example

Two examples are provided under `example/` folder. These are in `.ipynb` (notebook) format so as to show various plots with ease.
The notebooks were created in vscode 2019 and can be opened in IDEs with `.ipynb` supports.
A web-based IDE for python notebook is [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb#recent=true).

## Info

- Version: `0.1a`
- Python requirement: `>=3.6`
- Apr 30, 2020

## Reference

Ghahramani, Z., & Griffiths, T. L. (2006). Infinite latent feature models and the Indian buffet process. In Advances in neural information processing systems (pp. 475-482).

## License

GNU GENERAL PUBLIC LICENSE Version 3
