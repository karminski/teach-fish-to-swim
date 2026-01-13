One ruler to measure them all:
Benchmarking multilingual long-context language models
Yekyung KimÌ, Jenna RussellÌ, Marzena KarpinskaW, Mohit IyyerÌ,8
University of Maryland, College ParkÌ, MicrosoftW, UMass Amherst8
{yekyung, jennarus, miyyer}@umd.edu, mkarpinska@microsoft.com
Abstract
We present ONERULER, a multilingual benchmark designed to evaluate
long-context language models across 26 languages. ONERULER adapts the
English-only RULER benchmark (Hsieh et al., 2024) by including seven
synthetic tasks that test both retrieval and aggregation, including new variations of the “needle-in-a-haystack” task that allow for the possibility of a
nonexistent needle. We create ONERULER through a two-step process, first
writing English instructions for each task and then collaborating with native speakers to translate them into 25 additional languages. Experiments
with both open-weight and closed language models reveal a widening
performance gap between low- and high-resource languages as context
length increases from 8K to 128K tokens. Surprisingly, English is not the
top-performing language on long-context tasks (ranked 6th out of 26), with
Polish emerging as the top language. Our experiments also show that many
LLMs (particularly OpenAI’s o3-mini-high) incorrectly predict the absence
of an answer, even in high-resource languages. Finally, in cross-lingual
scenarios where instructions and context appear in different languages,
performance can fluctuate by up to 20% depending on the instruction language. We hope the release of ONERULER will facilitate future research into
improving multilingual and cross-lingual long-context training pipelines.
https://github.com/mungg/OneRuler
1 Introduction
Long-context language understanding is essential for real-world applications of large language models (LLMs) such as summarization and question answering. However, it is
difficult and expensive to conduct realistic evaluations for these tasks (Kim et al., 2024;
Karpinska et al., 2024), which motivates the use of synthetic benchmarks as proxy diagnostics. One popular example is the “needle-in-a-haystack” (NIAH) task (Kamradt, 2023),
in which a codeword is inserted into a long document and subsequently queried for. The
RULER benchmark (Hsieh et al., 2024) contains several variants of NIAH (e.g., multiple
needles and queries) as well as other synthetic tasks to test aggregation and variable tracing.
Unfortunately, RULER and other similar benchmarks mostly test long-context understanding
in either just English or in a small number of languages (Bai et al., 2024; Hengle et al.,
2024); as such, it remains unclear how well LLMs perform in multilingual and cross-lingual
long-context scenarios.
In this paper, we create ONERULER, a multilingual adaptation of RULER that includes seven
synthetic tasks (five variants of NIAH as well as two aggregation tasks) in 26 different
languages, including both low- and high- resource languages. While RULER is intended
to test base pretrained models, ONERULER is intentionally designed for models that have
been post-trained to follow instructions. Our data collection process involved first writing
instructions for all six tasks in English, and then hiring native speakers of the other 25
languages to translate these instructions. Unlike prior work, our NIAH instructions also
allow for the possibility of a nonexistent needle, where models get credit for identifying that
1
arXiv:2503.01996v3 [cs.CL] 30 Sep 2025
Published as a conference paper at COLM 2025
34%
23%
21%
11%
Top 5 Wikipedia
Bottom 5 Wikipedia
Resource type
Accuracy (%)
Context length
8k
32k
64k
128k
OneRuler: accuracy vs. context length on S-NIAH
20k 40k 60k 80k 100k 120k
90
80
70
60
50
40
100
(a) S-NIAH Avg. Accuracy
4%
32%
36%
8k
S-NIAH (without the none option)
S-NIAH (with the none option)
32k
64k
128k
20k 40k 60k 80k 100k 120k
100
90
80
70
60
40
Context length
Accuracy (%)
o3-mini S-NIAH performance with & without
none option in the instructions for English
50
30
(b) Impact of nonexistent needle on S-NIAH
Figure 1: (A) Micro-accuracy of all models on the S-NIAH task for the top 5 and bottom 5
languages by Wikipedia size. As context length increases, the performance gap between
high-resource and low-resource languages increases. (B) Performance of o3-mini-high on the
S-NIAH task in English, with and without the inclusion of the “None” option that allows for
the possibility of a nonexistent needle. Models are significantly more error-prone at longer
contexts when the prompt includes the possibility that the needle may not exist.
there is no answer. We show that this simple change dramatically lowers the performance
of models even on the vanilla NIAH task.
We benchmark four recently-released open-weight LLMs of different sizes, Qwen 2.5 (7B and
72B), Llama 3.1 (8B), and Llama 3.3 (70B), as well as two closed-source LLMs (OpenAI’s o3-
mini-high and Google’s Gemini 1.5 Flash). Overall, Gemini 1.5 Flash is the strongest tested
model in aggregate, followed by Qwen 2.5 72B; o3-mini-high, despite its powerful reasoning
capabilities, struggles badly on longer contexts. Interestingly, we observe a widening gap in
accuracy (averaged over all tasks and models) between low- and high-resource languages as
context length increases (Figure 1), suggesting a disparity between languages in long-context
pretraining and instruction tuning data.
Our experiments yield several surprising and counterintuitive results. For one, English is
not the highest-performing language across all models; in fact, it is the sixth-best language
out of the 26 when evaluated at long-context lengths (64k & 128k), while Polish takes the
top spot.1 Also surprising is the fact that even the vanilla NIAH task becomes challenging
when the prompt explicitly allows models to respond that the needle is absent, despite
near-perfect results observed in RULER and subsequent long-context LLM studies.2
In fact, a
large percentage of errors occur because models incorrectly decide that no needle exists.3
The most difficult task in ONERULER is the aggregation task, which requires listing the ten
most common words in a long list of words. Finally, in the cross-lingual setting, where the
instructions and context are in different languages, we observe that the accuracy can change
by up to 20% depending on the language of instructions.
2 Creating the ONERULER benchmark
ONERULER spans seven tasks adapted from RULER (Hsieh et al., 2024). Five are variants of
the needle-in-a-haystack retrieval task, differing in the number (and existence) of needles
and queries, while the other two require aggregating frequent words in a long list. For each
1Overall, the top-performing language families are Slavic, Romance, and Germanic, while Bantu
languages fare poorly despite having over 350M speakers.
2See e.g., Figure 2 of the Qwen 2.5 paper (Qwen Team, 2025), which shows a now-familiar bright
green rectangle exhibiting perfect NIAH performance.
3This result is reminiscent of the added challenge posed by SQuAD 2.0’s unanswerable questions
upon its release (Rajpurkar et al., 2018).
2
Published as a conference paper at COLM 2025
 The special magic number for
"forest" is: 2978103.
 The special magic number for
"table" is: 9128733.
 The special magic number for
"coffee" is: 1530998.
 The special magic number for
"apple" is: 7888231.
 The special magic number for
"garden" is: 4718002.
 The special magic number for
"queen" is: 6445721.
 The special magic number for
"queen" is: 4532661.
 The special magic number for
"queen" is: 3011363.
 The special magic number for
"queen" is: 5023114.
Multi-query NIAH
 The special magic number for
"river" is: 8923741.
 The special magic number for
"plane" is: 4225096.
 The special magic number for
"medicine" is: 9576844.
 The special magic number for
"newspaper" is: 1564287.
 The special magic number for
"paper" is: 8440387.
 The special magic number for
"island" is: 5132256.
1. advise 2. candidate 3. wetland 4. nosy 5. cop-out 6.
violation 7. toffee 8. itch 9. slavery 10. sensitive 11. decency
12. joyous 13. borrowing 14. tow-truck 15. condition 16.
packet 17. mover 18. shortage 19. sister 20. toffee 21. shift
22. jumbled 23. toffee ... 1622. ale 1623. imported 1624.
clogs 1625. vegetarianism 1626. pastoral 1627. equable 1628.
armor 1629. impress 1630. mainland 1631. monument 1632.
poem 1633. grab-bag 1634. toffee 1635. wetland 1636.
depression 1637. financing 1638. loose 1639. outfit 1640. big
1641. pastoral 1642. caper 1643. drain
<answer>2978103</answer> <answer>1530998</answer>
<answer>6445721, 4532661,
3011363, 5023114</answer>
<answer>8440387, 5132256</answer> <answer>none</answer> <answer>1. wetland, 2. pastoral,
..., 10. toffee</answer>
Figure 2: The seven tasks included in ONERULER. Spans highlighted in red are distractors,
while green spans contain answers that need to be produced for credit. CWE appears twice
(in easy and hard versions with differing frequencies) but shares the same format, hence
only one version is shown here. The NONE-NIAH task is a novel variant in which the
needle does not exist in the input context.
task, we evaluate four context lengths (8K, 32K, 64K, 128K) and 26 different languages, with
50 examples per configuration, totaling 5.2K prompts per task per model.
Languages: We include 26 diverse languages: Chinese (zh), Czech (cs), Danish (da), Dutch
(nl), English (en), Finnish (fi), French (fr), German (de), Hindi (hi), Hungarian (hu), Italian
(it), Japanese (ja), Korean (ko), Norwegian (no), Persian (fa), Polish (pl), Portuguese (pt),
Russian (ru), Serbian (sr), Sesotho (st), Spanish (es), Swahili (sw), Swedish (sv), Tamil (ta),
Ukrainian (uk), and Vietnamese (vi). These languages provide a solid representation of
different language families and writing systems (e.g., Latin, Cyrillic, logographic) and
exhibit a range of typological features, such as variations in word order and morphological
complexity. For fair comparison in retrieval and cross-lingual tasks, we also translated a
consistent set of 100 nouns into all 26 languages (see §A for more details).
High vs. low resource languages: Many of our experiments present comparisons between
high-resource and low-resource languages. To define what constitutes a low-resource language,
we rely on the official article count of Wikipedia articles per language (Joshi et al., 2020;
Ranathunga & de Silva, 2022; Nigatu et al., 2024),4 defining a minimum threshold of 250K
articles for a language to be considered high resource. Per this definition, we identify four
low-resource languages for our study: Hindi, Sesotho, Swahili, and Tamil.
Translating instructions: As an initial step, we translated English instructions and a list of
100 nouns into 25 languages. For 18 languages, we hired 17 Upwork annotators;5
for the
remaining 7 languages, we recruited 6 volunteers from the authors’ personal network. All
annotators were native speakers of the target languages with strong English proficiency.6
They were provided with context about the task and its objectives to ensure high-quality
translations. Annotators were instructed to translate and localize the instructions to make
4https://meta.wikimedia.org/wiki/List_of_Wikipedias
5https://www.upwork.com/
6Two annotators were native speakers of multiple languages and translated both of those languages
(Polish & Japanese, Russian & Ukrainian).
3
Published as a conference paper at COLM 2025
the prompts sound as natural as possible.7 They were also instructed to translate 100 nouns
based on provided definitions. After completing the initial translations, each annotator
reviewed the full set of instructions and made any necessary adjustments. Each annotator
was paid $25 USD per language to translate instructions and 100 nouns, totaling $492 USD.
8
Tokenization: It is difficult to conduct a fair comparison across models because they
use different tokenizers (Ahia et al., 2023): for example, one of our Tamil documents is
42,124 tokens using Gemini’s tokenizer and 103,990 tokens using Qwen’s tokenizer. This
discrepancy presents us with a choice of either (1) ensuring that the input text shown to each
model is identical, even if they have differing token counts across models; or (2) ensuring
that the total number of tokens shown to each model is identical, even if this means some
models see more text than others. We decide to report our main results using the second
setting to focus specifically on the effect of sequence length on model performance. However,
we also report results of experiments run under the first configuration in §D.
9
2.1 Retrieval tasks
We propose five retrieval tasks to assess the model’s ability to extract information from
extended contexts (see Figure 2). Each task is based on the needle-in-a-haystack paradigm
(Kamradt, 2023), where a target sentence is embedded within a longer text and the model
must retrieve specific details. Following RULER (Hsieh et al., 2024), we introduce three
variants that modify the number of needles and the amount of information to extract.
However, we deviate by reformatting all tasks for instruction-following models and also by
introducing the possibility of the answer not existing. When varying the needle’s position,
we make sure to follow each language’s spacing and punctuation conventions. To create
plausible contexts for needle injection, we collect and clean 26 open-domain non-copyrighted
books, one per language (see §B for more details). Each task is defined as follows:
• Single-NIAH (S-NIAH): This task follows the classic needle-in-a-haystack framework, where a single target sentence (the needle) is embedded in a book-length
context. The model must locate this sentence and retrieve the specific number
(“value”) associated with the keyword (“key”). In S-NIAH, only one needle is
present with no distractors. Unlike NIAH configurations in prior work, our prompt
template allows for the possibility of a nonexistent needle, even though the needle
always exists in S-NIAH. This decision (see bolded text in prompt below) reflects
real-world scenarios where questions may not always be answerable, and we ablate
its impact on performance in §4.
Please read and memorize the text below. I will ask you about it later.
<text> [CONTEXT] The special magic number for "[WORD 1]" is: [NUMBER 1]. [CONTEXT] </text>
<question> What special magic numbers associated with "[WORD 1]" are mentioned in the provided text?
Please list all that apply. If no such numbers exist, please answer "none". </question>
Please provide your answer in the following format: <answer>List all numbers here</answer>
• Multi-key NIAH (MK-NIAH): This variant embeds multiple needles with different
keys into the context; only one needle contains the correct key. Specifically, we insert
4 needles with unique keys, where 3 serve as distractors. The model must identify
the needle containing the target key and return its corresponding value.
• Multi-value NIAH (MV-NIAH): In contrast to MK-NIAH, this variant inserts 4
needles that share the same key but have different values. To successfully complete
the task, the model must retrieve all four values associated with the common key.
• Multi-query NIAH (MQ-NIAH): While sharing the same needle structure as MKNIAH, this variant presents multiple queries within each question. The model’s
response is considered correct only if it accurately retrieves all required information
7We pay special attention to the grammar of each language to ensure that any swap of variables
will not result in ungrammatical sentences.
8This cost includes contract and processing fees imposed by Upwork. The volunteers were not
paid for this task.
9We measure Kendall’s τ over the NIAH tasks across two settings and obtain a coefficient of 0.82
(p < 0.001), indicating strong agreement in model performance rankings.
4
Published as a conference paper at COLM 2025
Gemini 1.5 Flash Qwen2.5 72B o3-mini LLaMA 3.3 70B LLaMA 3.1 8B Qwen2.5 7B
8k 32k 64k 128k 8k 32k 64k 128k 8k 32k 64k 128k 8k 32k 64k 128k 8k 32k 64k 128k 8k 32k 64k 128k
Languages
ru
uk
fr
pl
it
es
fa
en
pt
sv
no
ja
de
hu
sr
cs
vi
fi
da
hi
zh
ko
ta
sw
st
nl Figure 3: Micro-accuracy across context-lengths and languages for all NIAH tasks. We
find that Romance languages perform best across all context lengths, along with Polish
and Russian. All models struggle on languages that use non-Latin scripts (except Cyrillic).
Gemini-1.5 Flash performs surprisingly well on Sesotho compared to other models.
for every query. This tests the model’s ability to maintain context awareness across
multiple retrieval operations.
• None-NIAH (NONE-NIAH): This novel variant tests a model’s ability to recognize
when no correct answer exists. The context contains four embedded needles that
all function as distractors. This challenges models to acknowledge the absence of a
correct response rather than forcing an incorrect selection. The prompt format is
identical to SINGLE-NIAH, but the correct answer is always absent.
2.2 Aggregation tasks
Unlike our retrieval tasks, which focus on extracting specific information from large and
irrelevant contexts, aggregation tasks require models to synthesize information across the
entire context to generate accurate responses. We adapt RULER’s Common Word Extraction
(CWE) task, which requires identifying the n most frequent words from a context (see §A
for more details). Our two CWE settings are:
• CWE-easy: The most frequent words in the list appear exactly 30 times each, while
other distractor words appear 3 times each. This replicates the parameters from
RULER, chosen because the task proves easy in short context settings but difficult in
longer contexts.
• CWE-hard: We also examine a more difficult setting that changes only the word
frequencies. In this setting, the most frequent words appear 20 times each while
distractor words appear 10 times each. This setting challenges models because of
the reduced frequency gap between answer words and distractors.
3 Experiments
We evaluate 7 different models on ONERULER across four context lengths, reporting accuracy
across models, languages, and tasks on the subset of returned responses.10 While most
10For the NIAH task, we discard no-answer cases (2.8% for o3-mini) and report micro accuracy over
the remaining instances. For CWE task, where such cases are more frequent (see §C.2), we treat them
as incorrect during evaluation.
5
Published as a conference paper at COLM 2025 Accuracy (%)
100
80
60
40
20
0
Models
Gemini-1.5-Flash
OneRuler accuracy per model for 64k & 128k
English
High-resource
Low-resource
Resource type
Qwen 2.5 72B o3-mini-high LLaMA 3.3 70B LLaMA 3.1 8B Qwen 2.5 7B
(a) Performance by model
fa fr es it en sv pt de ja nl no sr hu da hi cs fi zh ko ta sw st
Languages
100
80
20
0
OneRuler accuracy per language for 64k & 128kpl ru fr it es en uk sv pt de no nl hu da ja cs vi fi fa sr hi ko zhswtast
60
40
Accuracy (%)
EnglishHigh-resource
Low-resource
Resource type(b) Performance by language
Figure 4: NIAH performance across models and languages by language resource group for
long-context tasks (64K and 128K). Gemini 1.5 Flash demonstrates the best long-context
performance, while English and Chinese are surprisingly not among the top five languages.
models perform near perfectly on vanilla NIAH for English at short contexts (8k), accuracies
on low-resource languages and those that use non-Latin scripts is drastically lower, especially at longer context lengths. Only Gemini 1.5 Flash and Qwen 2.5 72B perform well on
NIAH tasks at long contexts (128K) on aggregate, but they still have room for improvement
especially on low resource languages. Our CWE aggregation tasks are difficult for all models,
especially the CWE-hard task: none of the test models achieves an accuracy above 1%.
Model selection: We evaluate 5 open-weights models: (DeepSeek-AI, 2025), Llama 3.3 70B
(Llama Team, 2024), Llama 3.1 7B (Llama Team, 2024), Qwen 2.5 (Qwen Team, 2025) in 7B
and 72B variants), and Deepseek-R1,11 the latter only for an analysis experiment in English.
We also compare to two closed-source models: Gemini 1.5 Flash and o3-mini-high. Notably,
Qwen was trained on 3T tokens of multilingual data with a particular focus on English and
Chinese. See §B.1 for more details on model configurations and resources.
3.1 Results
Figure 4b shows that ONERULER accuracy aggregated over all NIAH tasks and context
lengths is (unsurprisingly) higher for high-resource languages than low-resource languages.
We do see some correlation between model size and aggregate accuracy on low-resource languages, with the difference in accuracy between high and low resource languages shrinking
as model size increases (Figure 1a). We highlight several more interesting findings below:
The gap between high- and low-resource languages widens as context size increases: As
context size increases from 8K to 128K, Figure 1a shows that aggregate ONERULER accuracy
between the top five and bottom five languages by Wikipedia size widens considerably.
Specifically, the difference in aggregate accuracy increases from 11% with a context length of
8K to 34% with context length of 128K. We speculate that the widening gap might be due to
a lack of low-resource data used during long context extension (Gao et al., 2024; Lenz et al.,
2025; Llama Team, 2024): it is possible that long-context capabilities do not easily transfer
across languages.
Low-resource languages are challenging even at short contexts: All models demonstrate
strong aggregate ONERULER accuracy with a context length of 8K, as shown in Figure 4a.
However, they still struggle with low-resource languages like Swahili and Sesotho. This
issue is more pronounced in open models, with Llama models exhibiting the most severe
performance drops (see Figure 17). This is likely due to LlaMA being predominantly trained
on English-centric data (Llama Team, 2024); additionally, the inclusion of the nonexistent
needle negatively impacts NIAH task accuracy, as described later in §4.
11Although Deepseek-R1 is an open-weights model, it requires 8 H200-140GB GPUs for inference,
which exceeds our available resources. Therefore, we utilized the Fireworks API (https://fireworks.
ai/) for evaluation. Due to cost constraints, we limited our Deepseek-R1 experiments to English.
6
Published as a conference paper at COLM 2025
English and Chinese are not the highest-performing languages: English and Chinese
dominate the pretraining data of most modern languages, and so we might expect them
to be the top-performing languages on ONERULER. However, at context lengths of 64K and
128K, we unexpectedly observe that Polish is the top performer on NIAH tasks with an
average accuracy of 88% across all models, as depicted in Figure 4b. English is only the 6th
best language out of the 26, with an average NIAH accuracy of 83.9%. More shockingly,
Chinese is the 4th worst language on ONERULER, with an average NIAH accuracy of 62.1%.
While there seems to be some correlation between resource availability and performance
(all 4 low-resource languages rank in the bottom 6 languages), it remains unclear why some
high-resource languages like Chinese fare worse than anticipated.12 In contrast, the top 10
positions are occupied by Slavic, Romance, and Germanic languages, all of which have large
Wikipedia size (Figure 7) and use Latin scripts.
Individual model performance varies: Figure 3 displays the aggregate accuracy of different models on all ONERULER NIAH tasks as a function of language and context size.
While Gemini 1.5 Flash outperforms all other models across all context lengths, we observe
that Qwen 2.5 72B is consistently better than Llama 3.3 70B across all context lengths, with
notably higher performance in the 64k and 128k context-length settings. Also interesting is
the low average performance of o3-mini-high: it achieves only 67% accuracy on English at a
context length of 128K, compared to 92% on Polish and 89% on Ukrainian.
Figure 5: The performance of models on
each task, with bars representing English,
all other high-resource languages, and lowresource languages.
Models are surprisingly better on multiquery NIAH than single query NIAH for
languages other than English: Figure 5
presents task-wise performance. Surprisingly,
the models are better at retrieving two needles (MQ-NIAH) than one (S-NIAH). We
found that models tend to return ’none’ answers more frequently in S-NIAH than in
MQ-NIAH, leading to greater performance
degradation. We provide further analysis on
nonexistent needle in section 4. We also find
that MV-NIAH is more challenging than MKNIAH, possibly because models struggle to
retrieve all values associated with a single key
or terminate early. In addition, None-NIAH
exhibits the lowest performance among high-resource languages, suggesting that identifying
unanswerable cases remains the most difficult aspect of NIAH task.
CWE is much more challenging than NIAH: Compared to the NIAH tasks, on which
all models consistently achieve above 80% average accuracy on high-resource languages,
the CWE task presents a substantially greater challenge. Average English accuracy over all
models is only 31.5% for the CWE-easy task as shown in Figure 5.
13 Three models (Llama
3.3 70B, Qwen 2.5 72B, Gemini 1.5 Flash) achieve over 80% performance at 8K context, but
performance drops drastically as context length increases. The CWE-hard setting proves
unsolvable with nearly 0% accuracy across all models, indicating that LLMs have significant
room for improvement on long-context aggregation tasks. We further analyze performance
across context lengths and models in §C.3.
12We observe that Qwen’s errors on the Chinese S-NIAH task are primarily due to the model
frequently generating incorrect ’none’ responses. This type of error is not unique to Qwen; it also
appears across other models, most notably in o3-mini-high, which exhibits a significant number of
such wrong answers (see §4).
13We note that 4 languages (ko, zh, st, sw) have contexts shorter than 128k tokens because the
required number of words exceeded our available vocabulary size.
7
Published as a conference paper at COLM 2025
Context language
en ko
ko
pl
pl en
en ko pl
en ko pl en ko pl
ko pl en
ko pl en ko pl en
Instruction language
8k 32k
64k 128k
(a) Cross-lingual Performance
other models: none errors
o3-mini: remaining errors
other models: remaining errors
8k 32k 64k
o3-mini: none errors
128k
S-NIAH error types: o3-mini vs. other models
20
15
10
5
0
Average error count
40.7%
59.3%
93.2% 57.6%
42.4%
90.2%
9.8%
53.3%
46.7%
11.9%
88.1%
46.2%
53.8%
Context length
(b) o3-mini-high Errors in S-NIAH task
Figure 6: (A) The cross-lingual average accuracy of En, Ko, an Pl on NIAH tasks at each
context length. We find the language of instruction can make a significant impact on overall
model performance. (B) The types of errors made in the S-NIAH by o3-mini-high vs other
models tested. o3-mini-high is more likely to generate an errors than other tasks, and is
much more likely to answer ‘none’, despite an answer being present.
4 Analysis
In this section, we dig into some of the surprising results we observe above, seeking to
understand what properties of the tasks in ONERULER most trouble the models we tested (e.g.,
nonexistent needles, inefficient reasoning and language-specific issues). We also explore a
cross-lingual setting in which task instructions and input context are in different languages.
The option to answer none makes NIAH significantly harder: Since tasks like NoneNIAH inherently lack valid answers, we explicitly provided an option for models to respond
accordingly by including the instruction: If no such number exists, please answer ‘none‘ (Figure 9). This simple addition made our NIAH tasks much harder than those in RULER:
Figure 1b shows that adding this sentence drops S-NIAH accuracy by 32% at a context
length of 128K in English. We observe several models, and in particular o3-mini-high as
shown in Figure 6b, have a common failure mode of responding none when the needle actually exists in the context (see Figure 18 and Figure 19 for more detailed analysis). We suspect
the inclusion of this sentence may make models overly cautious to responding, and/or
many of these models include NIAH data (without the ‘none‘ option) during post-training.
Reasoning models behave strangely on NIAH tasks: Interestingly, we observe that o3-
mini-high produces significantly more reasoning tokens for its incorrect answers than
for its correct answers (see Figure 20), suggesting that its reasoning behavior for these
simple retrieval tasks is highly inefficient (Sui et al., 2025). We also run Deepseek-R1 on
the English set of ONERULER to see whether it behaves similarly; unfortunately, its high
overall performance on English tasks prevents meaningful analysis of reasoning length
on incorrect examples. Instead, we examine its visible reasoning traces, which reveal an
interesting approach to NIAH tasks. Some reasoning examples from Deepseek-R1 show
that the model adopts a systematic strategy where it divides the context by chapters and
simultaneously performs both summarization and targeted searches for magic numbers.
See §C for a detailed example.
Cross-lingual performance is language-dependent: To test cross-lingual long-context
capabilities, we run a variation of ONERULER’s NIAH tasks where the instruction language
differs from the language of the context and the needles. We run experiments on English,
Polish, and Korean (Figure 6a), finding that the choice of instruction language significantly
impacts performance. Given English contexts, switching the instruction language to a lower
resource language such as Korean decreases the average NIAH accuracy across all models
8
Published as a conference paper at COLM 2025
LANG TASK MODEL EXAMPLE COMMENT
DE NIAH
Single
Qwen 2.5 7B <Antwort> 1600 </Antwort> Die Zahl 1600 wird im Text erwähnt in Bezug auf
die Höhe des Ortes, an dem Joachim Ziemßen lebt: "Sechzehnhundert Meter über
dem Meer".
Translation: The number 1600 is mentioned in the text in relation to the elevation
of the place where Joachim Ziemßen lives: "sixteen hundred meters above sea
level."
Used context instead of
needle to create answer.
ZH NIAH
Single
o3-mini
(high)
<答案>2</答案> 解析：在文本中有一道：“我史一. . . . . . 本都着字是‘吃人’！上着多
字. . . . . . ”里“”上所的容是“吃人”字，因此“”相的魔法字就是“2”。
Translation: <Answer>2</Answer> Explanation: In the text, there is a passage
that reads, "I flipped through history... the entire book was filled with two characters: ’eat people’! The book contained so many words..." Here, the content written
on the "book" consists of the two characters for "eat people," so the magical number associated with the book is "2."
Model unnecessarily
complicates the task
where it simply has to
extract the magic number
associated with “book”
but instead creates its
own riddle.
PL NIAH
None
Qwen 2.5 7B <odpowied´z> Brak </od 프로그래머: 文字示不正常，可能是。保使用UTF-8打文件。
Translation: <Answer> None </AnProgrammer: The text is not displaying correctly, possibly due to encoding issues. Please open the file using UTF-8 encoding.
Model answers then
starts producing Korean
and switches into Chinese.
JA CWE easy o3-mini
(high)
<答え> 在る; 有る, 議室, けがらわしい, 組合せ, もらう, 出る, 社, 連休, 年,
物 </答え>
※なお、こちらの回答は投稿の全語リストを機械的に頻度解析した結果（想的な解析
例）に基づくものであり、際のデータ理結果と異なる可能性があります。
Translation: <Answer> Exist; Have, Conference Room, Filthy, Combination,
Receive, Appear, Company, Consecutive Holidays, Year, Object </Answer>
※ This answer is based on a mechanical frequency analysis of all words in the
post (a hypothetical analysis example) and may differ from actual data processing
results.
Model produces a hypothetical answer.
EN CWE easy Llama 3.1 8B <Answer> 1. the 2. and 3. of 4. to 5. a 6. in 7. that 8. is 9. for 10. with </Answer> Model returns generic
high-frequency words
unrelated to the context.
Table 1: Example failures for both NIAH and CWE tasks. Other common cases include
returning incomplete answers, partially correct answers, or ‘none’ when the answer exists.
at all context lengths; for example at 64K context-length performance drops from 91% to
71%. However, if the context is in Korean, switching the instructions to English or Polish
improves performance: for example, at a context length of 128K, average accuracy increases
from 61% to 77% when instructions are switched from Korean to English. Taken as a whole,
our preliminary study forms a starting point for cross-lingual long-context benchmarking
of different training and data generation strategies.
Complications with CWE: CWE requires models to correctly identify all 10 common
words, a task that is trivial for humans but remains surprisingly challenging for LLMs. In
the easy setting, models often return 8–9 correct words, while in the hard setting, most fail
entirely. High-resource languages tend to perform slightly better, but this advantage diminishes as context length increases (Figure 23). Notably, the list-of-words format used in CWE
makes the task especially sensitive to tokenization. In multilingual settings, tokenizers that
produce fewer tokens (e.g., o200k in o3-mini and Gemini) result in a much larger candidate
word pool for some languages as shown in Figure 24, complicating fair comparison across
languages. Additionally, reasoning models such as o3-mini-high and Deepseek-R1 often
exceed their output token limits (Figure 21). This is largely due to their tendency to recall
word lists verbosely. In summary, CWE highlights both model limitations and structural
challenges in multilingual evaluation. This motivates future work on bits-per-byte style
normalization for multilingual evaluation.
Analysis of common errors: For the S-NIAH, models frequently answer ‘none’ (see
Figure 6b). Other NIAH tasks are affected by this at lower frequencies, sometimes with
numerical responses provided alongside ‘none’ especially if more than one value was
requested. In multi-key and none NIAH, models often return distractors. In multi-query
NIAH, they typically produce only one needle instead of the required two. Similarly, in
multi-value NIAH, models often miss at least one of four values. Llama and Qwen models
fall into loopy number repetitions, sometimes incrementing them by one, a failure more
common in their smaller variants. In CWE tasks, models frequently return only a subset
of the top 10 words, with accuracy declining as context length increases (see Figure 23).
Furthermore, a performance gap exists between high and low-resource languages at shorter
context lengths, but it narrows at longer contexts where both perform poorly. Finally, we
9
Published as a conference paper at COLM 2025
observe models either hallucinating answers, reformulating the task, or, in the case of Qwen
2.5 7B and LlaMa 3.1 8B, mixing languages almost exclusively for Polish (see Table 1).
5 Related work
Evaluation of multilingual long-context LLMs: Most related to our work are prior
efforts to benchmark multilingual long-context language models. LongBench (Bai et al.,
2024) includes both synthetic and natural tasks in English and Chinese, while Tanzer et al.
(2024) evaluates language models’ ability to translate from English to Kalamang, a lowresource language with under 200 speakers. There are also several multilingual variants of
NIAH (Hengle et al., 2024; Agrawal et al., 2024; Huang et al., 2025); however, ONERULER
includes many more languages than these efforts, in addition to the none answer type and
evaluation of reasoning models.
Synthetic long-context benchmarks: We build on prior synthetic evaluations, most
notably RULER (Hsieh et al., 2024), to benchmark of long-context LLM capabilities. Most of
these are largely based on the “needle-in-a-haystack” framework (Kamradt, 2023), which has
gained popularity due to its ease of evaluation and modification (Yuan et al., 2024; Xu et al.,
2024; Song et al., 2025; Laban et al., 2024; Sharma et al., 2024). Outside of NIAH, the recent
LongReason benchmark (Ling et al., 2025) expands the context of short-context reasoning
questions to evaluate long-context capabilities, while GSM-∞(Zhou et al., 2025) generates
long-context tasks with controllable complexity and information density via computational
graphs.
Realistic long-context benchmarks: While synthetic tasks are cheap and easy to control,
they also do not test real-world tasks; as such, other benchmarks (mostly in English) focus
on specific tasks such as QA (An et al., 2024; Levy et al., 2024), summarization (Kim et al.,
2024) or a suite of many realistic tasks (Shaham et al., 2023; Dong et al., 2023; Li et al., 2024;
Lee et al., 2024; Yen et al., 2025). InfiniteBench (Zhang et al., 2024) pushed evaluation of
context lengths past 100K tokens. Others have proposed evaluation of real-world tasks such
as conversations with agents (Castillo et al., 2024), and code understanding (Liu et al., 2024).
BABILong (Kuratov et al., 2024) and NoCha (Karpinska et al., 2024) both evaluate reasoning
of factuality over long contexts.
6 Conclusion
We introduce ONERULER, a synthetic benchmark for multilingual long-context language
models across 26 languages that measures both retrieval and aggregation capabilities. Our
experiments reveal that performance disparities between high- and low-resource languages
increase as context length increases. We hypothesize these performance differences stem
from factors such as pretraining data availability, script, language family, and tokenizer specifications. Contrary to expectations, English and Chinese are not among the top-performing
languages, with Polish taking the top spot. Furthermore, we observe that introducing the
possibility of nonexistent needles sharply decreases NIAH performance on all models. We
release ONERULER to spur the development of multilingual long-context LLM capabilities.
Acknowledgments
We would like to extend our gratitude to the Upwork annotators for their dedicated efforts,
and to Ankita Gupta, Chau Pham, Rishanth Rajendhran, Yixiao Song and for voluntarily
contributing to the translation. We are also grateful to the members of the UMass NLP and
UMD CLIP labs for their valuable feedback. Our deep appreciation goes to Simeng Sun,
whose discussions inspired the initial concept of this work and provided many valuable
insights. This project was partially supported by awards IIS-2046248, IIS-2312949, and
IIS-2202506 from the National Science Foundation (NSF).
