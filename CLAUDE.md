# Auto research using Amazon SageMaker

## Goal

I want to create an auto research tool to run ML experimentation loops in the SageMaker platform.

## Background

Auto research is a variant of loop engineering. We start with a goal, like build a classification model that achieves an F1 score of 0.9 or higher, and let our coding agent continue to work until it hits the goal. 

The original auto research repository was at https://github.com/karpathy/autoresearch. There's newer versions like https://github.com/uditgoenka/autoresearch. 

These are general purpose auto research harnesses.

## Auto research for ML

We want to produce a version of an auto research harness tailored for working with Amazon SageMaker. It should cover the full ML lifecycle, from data preparation to feature engineering to running a training job and capturing experimental results.

* Assume our data is available in Amazon Redshift.
* The user must specify which data can be used. The auto research tool can select from available data and figure out the best data preparation and mixing options.
* The user must specify the success criteria, like F1 score >= 0.9.
* The auto research tool should do feature engineering using SageMaker jobs if possible.
* The auto research tool should use SageMaker's MLFlow server to capture all experiments.
* The auto research tool should use SageMaker to run ML training jobs.
* The auto research tool should stop when it has a candidate model, registered in the SageMaker model catalog, that achieves the goal.
* The auto research tool will inherit access to the user's AWS account through an AWS profile.

## Output

The output of this project is a Claude Code "SageMaker Auto Research" plugin.
