The folder `syntax-twins/` contains, in separated files, a set of 2098 original sentences $\mathbf{X}_i$, their syntax twins $\mathbf{s}^0_i$, and their (mutual) POS_templates.

The folder `semantics/` contains, in separated txt files, a set of 2018 original sentences $\mathbf{X}_i$, their paraphrases $\mathbf{P}_i$, and their translations into each one of the languages we used.

Due to different filters and cleaning procedures applied, 1584 original sentences are simultaneously present in both 
`syntax-twins/original_sentences_syntax.txt` and 
`semantics/original_sentences_semantics.txt`.

The jupyter notebook `matching_data.ipynb` finds the shared original sentences between 
`semantics/original_sentences_semantics.txt`
and<br>
`syntax-twins/original_sentences_syntax.txt`,
and their positions in each file.



