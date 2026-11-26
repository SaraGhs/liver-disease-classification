# Personal Contribution and Interpretation

## Project context

This portfolio refresh builds on my earlier Machine Learning for Bioinformatics coursework. The earlier course exercise and the current documented rerun are separate stages. The refreshed workflow and starter documentation were prepared with AI assistance.

## My contribution to the current version

I ran the supplied analysis notebook in Google Colab and exported the analysis script, environment details, split assignments, cross-validation results, held-out predictions, model outputs and figures for review. I am reviewing the methods and their interpretation with assistance before finalizing the portfolio documentation.

The preprocessing design, grouped splitting, hyperparameter search, automated model selection and plotting were implemented in the starter code. I do not claim to have independently designed or written every component of this refreshed workflow. This statement records the supported execution and reporting contribution; it does not attribute additional code changes to me without documentation.

## What the workflow does

The analysis uses ten input features to predict the dataset's liver-disease label. It excludes the target from the inputs, removes exact duplicate records, reserves a held-out partition and fits preprocessing only within training folds. It compares a majority-class baseline, logistic regression and an RBF SVM.

The target must not be an input because it is the answer the model is supposed to predict. Including it would let the model use information unavailable when predicting a genuinely unknown label. Imputation and scaling are learned from training records so that validation and test data do not influence the fitted preprocessing steps.

## Interpretation of this run

Logistic regression was selected using training cross-validation, with a mean balanced accuracy of 70.75%. Its held-out balanced accuracy was 72.65% and its disease recall was 56.41%.

On the 114-record test partition, it correctly classified 44 disease-labelled records and 32 no-disease records. It missed 34 disease-labelled records and produced four false positives. Its 91.67% disease precision describes the correctness of its positive predictions, not the fraction of all disease-labelled records detected.

The majority baseline achieved 68.42% ordinary accuracy by predicting disease for every test record. This illustrates why overall accuracy alone is insufficient for comparing these classifiers. Although logistic regression improved balanced accuracy over the baseline, its ordinary accuracy was lower and its false-negative count is an important limitation.

The SVM achieved slightly higher held-out balanced accuracy, but the selected model remains logistic regression because selection was based on training cross-validation. Switching to SVM after inspecting the test results would change the role of that test set.

## Scope and next steps

This is a small educational comparison, not a validated diagnostic tool. Further work could include a prespecified investigation of the recall/false-positive trade-off using training-only validation, uncertainty estimates and external data if available. Those extensions have not been completed and are not claimed as current results.
