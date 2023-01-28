# ConversationRecSys
## Background
In traditional recommend systems, models use historical user data to recommend from one side.
However, conversational recommend systems show advantages in scenarios like cold-start and communications with the disabled.
Based on the reference, we re-implemented and improved a conversational recommend system that supports natural language interaction between users and the system.

## Challenges
The conversation itself does not contain enough background information to understand users' preferences.
What's more, natural language interaction has a wide variety among individuals.

## Contribution
Led a three-person team and implemented the natural langage generation model and semantic fusion/loss function in the training part. Improved the generation language quality from *Distinct 0.6204* to *Distinct 0.9304*.

## Setup
Python3.7
### OS X

torch                     1.3.0(post2)             pypi_0    pypi

torch-cluster             1.4.5                    pypi_0    pypi

torch-geometric           1.3.2                    pypi_0    pypi

torch-scatter             1.3.2                    pypi_0    pypi

torch-sparse              0.4.3                    pypi_0    pypi

torchvision               0.4.1                    pypi_0    pypi

No CUDA

### Windows
CUDA9.2

source：https://download.pytorch.org/whl/torch_stable.html

## Training
- Pretraining
(Semantic fusion and recommend task)
python run.py

- Fine-tuning
（Conversation task）
python run.py --is_finetune=True

## Output
*test_context.txt*	:		Conversation data for validation & testing in each epoch (90 in total)

*test_output.txt* :		Model-generationed response

## Evaluation
![After improvement](https://github.com/Irreel/ConversationRecSys/blob/main/eval.png)

Language quality *Distinct*: 0.9304

## Reference
Zhou, K., Zhao, W. X., Bian, S., Zhou, Y., Wen, J. R., & Yu, J. (2020). Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1006–1014. https://doi.org/10.1145/3394486.3403143
