# BlyxTorch independent verification

These tools compare the Blyx implementation with CPU PyTorch and exercise the distributed command-line interface. Framework users can install and run BlyxTorch independently of this repository.

Use Python 3.11 or later and the Blyx 0.2.1 compiler. Build the framework first, then pass its executable to the checks. For the reference comparison and timing harness, install CPU PyTorch and NumPy in a separate environment.

~~~sh
python test_cli.py --source ../blyxtorch --exe ../blyxtorch/build/blyxtorch
python pytorch_reference.py --exe ../blyxtorch/build/blyxtorch
python bench_score.py --exe ../blyxtorch/build/blyxtorch
~~~

The CLI checks exercise deterministic training, CSV validation, generated models and checkpoint round trips. The numerical comparison uses PyTorch autograd and selected central differences. The benchmark reports warm and fresh-process timings separately and records the executable's version. The native Blyx test suite lives with the framework. [BlyxTorch source](https://github.com/Rahulchaube1/blyxtorch)

## Connect with Rahul Chaube

[Portfolio](https://www.rahulchaube1.com/) · [GitHub](https://github.com/Rahulchaube1) · [LinkedIn](https://www.linkedin.com/in/rahulchaube1) · [Hugging Face](https://huggingface.co/oncody) · [X](https://x.com/RahulChaube_) · [EverestQ](https://www.everestqai.com/) · [Contact](https://www.rahulchaube1.com/contact)

[Blyx website](https://www.blyx-lang.space/) · [BlyxTorch](https://www.blyx-lang.space/packages/blyxtorch) · [Handbook](https://doc.blyx-lang.space/) · [Community](https://www.blyx-lang.space/community)

Support Rahul’s open-source work through [GitHub Sponsors](https://github.com/sponsors/Rahulchaube1) or [Buy Me a Coffee](https://buymeacoffee.com/rahulchaube).
