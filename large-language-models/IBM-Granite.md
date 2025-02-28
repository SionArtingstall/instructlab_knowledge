IBM Granite is a series of decoder-only AI foundation models created by IBM. It was announced on September 7, 2023,and an initial paper was published 4 days later.[6] Initially intended for use in the IBM's cloud-based data and generative AI platform Watsonx along with other models,[7] IBM opened the source code of some code models. Granite models are trained on datasets curated from Internet, academic publishings, code datasets, legal and finance documents.

# Foundation models
A foundation model is an AI model trained on broad data at scale such that it can be adapted to a wide range of downstream tasks.

Granite's first foundation models were Granite.13b.instruct and Granite.13b.chat. The "13b" in their name comes from 13 billion, the amount of parameters they have as models, lesser than most of the larger models of the time. Later models vary from 3 to 34 billion parameters.

On May 6, 2024, IBM released the source code of four variations of Granite Code Models under Apache 2, an open source permissive license that allows completely free use, modification and sharing of the software, and put them on Hugging Face for public use. According to IBM's own report, Granite 8b outperforms Llama 3 on several coding related tasks within similar range of parameters.

# Granite 3.2 models
All Granite 3.2 models are available under the permissive Apache 2.0 license on Hugging Face. Select models are available today on IBM watsonx.ai, Ollama, Replicate, and LM Studio, and expected soon in RHEL AI 1.5 – bringing advanced capabilities to businesses and the open-source community.


The new Granite 3.2 8B Instruct and Granite 3.2 2B Instruct offer experimental chain-of-thought reasoning capabilities that significantly improve their ability to follow complex instructions with no sacrifice to general performance. The reasoning process can be toggled on and off, allowing for efficient use of computing resources.
When combined with IBM’s inference scaling techniques, Granite 3.2 8B Instruct’s extended thought process enables it to meet or exceed the reasoning performance of much larger models, including GPT-4o and Claude 3.5 Sonnet.
Our new multimodal model, Granite Vision 3.2 2B, was developed with a particular focus on document understanding, on which it matches the performance prominent open models 5 times its size.
The latest additions to the Granite Timeseries model family, Granite-Timeseries-TTM-R2.1, expand TTM’s forecasting capabilities to include daily and weekly predictions in addition to the minutely and hourly forecasting tasks already supported by prior TTM models.
We’re introducing new model sizes for Granite Guardian 3.2, including a variant derived from our 3B-A800M mixture of experts (MoE) language model. The new models offer increased efficiency with minimal loss in performance.
The Granite Embedding model series now includes the ability to learn sparse embeddings. Granite-Embedding-30M-Sparse balances efficiency and scalability across diverse resource and latency budgets.
Like their predecessors, all new IBM Granite models are released open sourced under a permissive Apache 2.0 license.
Granite 3.2 models are now available on IBM watsonx.ai, Hugging Face, Ollama, LMStudio, and Replicate.

Granite 3.2, the latest release in our third generation of IBM Granite models, is an essential step in the evolution of the Granite series beyond straightforward language models. Headlined by experimental reasoning features and our first official vision language model (VLM), Granite 3.2 introduces several significant new capabilities to the Granite family.

The release also includes an array of improvements to the efficiency, efficacy and versatility of our existing offerings. IBM’s prioritization of practical, enterprise-ready models continues the pursuit of state-of-the-art performance with fewer and fewer parameters.

# Granite 3.2 Instruct: Reasoning when you need it
The newest iterations of IBM’s flagship text-only large language models (LLMs), Granite 3.2 Instruct 8B and Granite 3.2 Instruct 2B, have been trained to offer enhanced reasoning capabilities relative to their 3.1 counterparts. Our implementation of reasoning runs somewhat counter to certain industry trends, in keeping with IBM’s practical approach to enhancing model performance.

Rather than complicating development pipelines by releasing separate “reasoning models,” IBM has baked reasoning capabilities directly into our core Instruct models. The model’s internal reasoning process can be easily toggled on and off, ensuring the appropriate use of compute resources for the task at hand.

Whereas typical reasoning-driven techniques improve model performance on logical tasks (such as math and coding) at the expense of other domains, IBM’s methodology brings the benefits of reasoning while preserving general performance and safety across the board.
These experimental features of the new Granite 3.2 Instruct models represent only one of multiple ongoing explorations at IBM Research into reasoning-driven model evolution. Further work on inference scaling techniques demonstrates that Granite 3.2 8B Instruct can be calibrated to match or exceed the mathematical reasoning performance of much larger models, including OpenAI’s GPT-4o-0513 and Anthropic’s Claude-3.5-Sonnet-1022.

Addressing the advantages (and disadvantages) of reasoning
The intuition driving recent advancements in language model reasoning comes from 2022 research demonstrating that simply adding the phrase “think step by step,” a prompt engineering technique commonly called chain of thought (CoT) prompting, significantly improves model outputs on reasoning tasks.1

Subsequent research from 2024 further posited that scaling up inference-time compute—that is, the resources used to generate each output during inference—could enhance model performance as much as scaling up the size of a model or resources used to train it. The most recent approaches have mostly pursued such inference scaling through the incorporation of various reinforcement learning (RL) frameworks that incentivize longer, more complex “thought processes.” Excitingly, inference scaling has been empirically demonstrated to enable even smaller LLMs to exceed the reasoning abilities of much larger models.

Despite their strengths, reasoning models are not without downsides. Understanding this, IBM took deliberate measures to mitigate these disadvantages in the specific implementation of reasoning capabilities for Granite 3.2.
Avoiding inefficiency
“Reasoning models” are typically slower and more expensive than general LLMs, since you must generate (and pay for) all the tokens the model uses to “think” about the final response before actually providing an output back to the user. IBM Research noted one example of DeepSeek-R1, a prominent reasoning model, taking 50.9 seconds to answer the question, “Where is Rome?”

There are scenarios in which that extra time and compute can be easily justified, but there are also many scenarios in which it becomes a waste of resources. Rather than requiring developers to juggle these tradeoffs each time they choose a model for a given application or workflow, IBM Granite 3.2 Instruct models allow their extended thought process to be toggled on or off by simply adding the parameter  "thinking":true or "thinking":false to the API endpoint. 

You can tap into Granite 3.2’s thought process when it’s necessary or prioritize efficiency when it isn’t.

# Avoiding general performance drops
In the relatively short history of reasoning models, many prominent approaches have prioritized performance gains on only a narrowly-focused set of logic-driven domains, such as math or coding. While IBM’s ongoing work with inference scaling techniques has yielded particularly impressive performance improvements on technical benchmarks conventionally associated with “reasoning,” such as AIME and MATH-500, our focus for Granite 3.2 Instruct was on enriching our models’ thought processes to more broadly improve their ability to follow complex instructions.

A narrow focus on technical tasks explicitly targeted by the model developers can sometimes be at the expense of other domains—including general performance and safety—whose knowledge can be “forgotten” by the model if they’re not adequately covered in the datasets used to improve reasoning performance. To avoid this, IBM developed Granite 3.2 Instruct by applying a Thought Preference Optimization (TPO)-based reinforcement learning framework to directly to Granite 3.1 Instruct.

Unlike many common approaches to reasoning capabilities, TPO's lesser reliance on logical operators or functions to rate and reward model outputs makes it easier to scale to general tasks. This enabled Granite 3.2 Instruct to enjoy increased performance on tasks requiring complex reasoning without compromising performance elsewhere.

The benefits of this approach are most evident in comparisons against DeepSeek-R1-Distill models, which (despite their names) are actually versions of Llama and Qwen models fine-tuned to emulate DeepSeek-R1’s reasoning process. It’s worth noting here that, unlike the R1-Distill models, IBM Granite 3.2 Instruct models were not trained using any DeepSeek-generated data, greatly simplifying their regulatory implications.

Consider the pre- and post-reasoning performance of similarly sized Llama, Qwen and Granite models on ArenaHard and Alpaca-Eval-2, popular benchmarks that measure a model’s ability to think their way through difficult instructions. Whereas DeepSeek’s technique decreases performance on these non-targeted tasks, the CoT techniques used to evolve Granite 3.1 Instruct into Granite 3.2 Instruct significantly improved instruction-following.

# For Enterprise
IBM keeps enterprise-essential concerns, including safety, at the heart of all design decisions. While the DeepSeek-distilled models show a significant drop in safety performance (as measured by performance on the AttaQ benchmark), IBM’s approach preserved Granite 3.2 Instruct’s robustness to adversarial attacks.
Continuing our work on reasoning
As mentioned, the release of Granite 3.2 marks only the beginning of IBM’s explorations into reasoning capabilities for enterprise models. Much of our ongoing research aims to take advantage of the inherently longer, more robust thought process of Granite 3.2 for further model optimization.

One such avenue of exploration is centered around bolstering Granite 3.2 with more complex inference scaling techniques, including particle filtering and majority voting (also called self-consistency). Early experiments demonstrate that, when used in conjunction with these inference scaling techniques, Granite 3.2’s performance on mathematical reasoning tasks can match or exceed the performance of much larger frontier models.

# Granite Vision 3.2 2B: Granite goes multimodal
Granite Vision 3.2 2B is a lightweight large language model with computer vision capabilities that target everyday enterprise use cases, trained with a particular focus on visual document understanding. Handling both image and text inputs, Granite Vision 3.2's performance on essential enterprise benchmarks, such as DocVQA and ChartQA, rivals that of even significantly larger open models.

While Granite Vision 3.2 2B is not explicitly intended to be a drop-in replacement for similarly sized text-only Granite models on language tasks, it can capably handle text-in, text-out scenarios.

# Vision with an eye for enterprise images
Granite Vision 3.2 2B can handle a wide variety of visual understanding tasks, but it specializes in tasks most relevant to document understanding and multimodal retrieval augmented generation (RAG).

Most VLMs, alternatively called multimodal large language models (MLLMs), are trained for vision tasks predominately on natural images. This does not necessarily yield optimal performance on images of documents, whose unique visual characteristics—layouts, fonts, charts, infographics—differ significantly from those of natural images. Relative to most generalized image-in, text-out use cases, document understanding requires a more specific and fine-grained comprehension of visual context.

The two primary challenges in enabling MLLMs to effectively process documents and associated visuals are adequately encoding high-resolution images and accurately interpreting visually-situated text within those documents. Specialized approaches typically either rely on external optical character recognition (OCR) systems to process text within images in a “perceive-then-understand” framework or bespoke model architectures designed exclusively for document understanding.

Both approaches have drawbacks. Dependence on external OCR-driven document understanding can result in the accumulation of errors prior to essential information reaching the language, while many dedicated “OCR-free” methods struggle to handle high-resolution input or suffer from a lack of overall knowledge relative to that of a competitive LLM.2

More recently, strong performance in document understanding has been achieved by instruction-tuning generalized vision language models on document-focused datasets. Unfortunately, progress in this approach has been somewhat limited by a shortage of suitable open source datasets. To facilitate further progress with this approach, IBM’s development of Granite Vision 3.2 involved extensive work toward a comprehensive instruction-following dataset for visual document understanding.

Sparse attention vectors for intrinsic safety monitoring
In the design and training of Granite 3.2 Vision, IBM also introduced a novel test-time technique that, rather than relying on an external guardrail model to monitor harmful activity, incorporates a dedicated safety approach directly into the model itself.

Our key insight is that within Granite Vision’s many attention heads and transformer layers is a sparse subset of image features that could be useful for identifying safety concerns when safety monitoring tasks are formalized as classification problems.

In a process detailed further in the Granite Vision technical paper, IBM Research designed a process to isolate and examine the attention vectors produced within Granite Vision’s attention mechanism in order to evaluate which, on average, reliably correlate with certain classes of harmful inputs. Once identified, the attention heads responsible for generating those “safety vectors” can be used to determine whether a given input is safe.

IBM will continue to explore the potential applications of sparse attention vectors. One potential avenue of exploration investigate their use in adapting future versions of Granite Guardian for fully multimodal safety monitoring.

# Granite Guardian 3.2 
The latest generation of IBM guardrail models designed to detect risks in prompts and responses, provides performance on par with Guardian 3.1 equivalents at greater speed with lower inference costs and memory usage.

Verbalized confidence
IBM Granite Guardian 3.2 introduces verbalized confidence, a new feature that provides a more nuanced evaluation of detected risks to acknowledge the ambiguity inherent to certain safety monitoring scenarios.

Rather than solely outputting a binary “Yes” or “No” in the process of monitoring inputs and outputs for risk, Granite Guardian 3.2 models will also indicate their relative level of certainty. When potential risks are detected, Guardian 3.2 models indicate either “High” or “Low” confidence, as demonstrated in the following example:

label, confidence = parse_output(output, input_len)

print(f"# risk detected? : {label}") # Yes

print(f"# confidence detected? : {confidence}") # High

Granite Guardian 3.2 introduces two new model sizes:

Granite Guardian 3.2 5B was derived from Guardian Guardian 3.1 8B (which itself was created through fine-tuning the base language model for safety classification). Inspired by research demonstrating that the deeper layers of a neural are often either redundant, not fully leveraged by pretraining or simply less critical than the networks’ shallower layers, IBM pursued an iterative pruning strategy to “thin” the 8B model. The process resulted in a roughly 30% reduction of the 8B’s parameters while retaining performance close to that of the original model.

First, specific layers for pruning are selected based on the relative similarity between their input vectors and output vectors. In other words, we identify the network layers whose contributions are least impactful.
Once identified, 10 layers are eliminated from the model.
The model is then “healed” by retraining it on 80% of the original training data, after which 2 more layers are pruned.
Granite Guardian 3.2 3B-A800M was created by fine-tuning our mixture of experts (MoE) base model, which activates only 800M of its 3B total parameter count at inference time. Its introduction adds an especially efficient and cost-effective option to the Granite Guardian lineup.

Granite Timeseries models: Now with daily and weekly forecasting
IBM’s popular open source family of compact Granite Time Series models, dubbed Tiny Time Mixers (TTMs), have been downloaded over 8 million times on Hugging Face. While prior TTM variants released within the TTM-R1 and TTM-R2 series supported zero-shot and few-shot forecasting for minutely to hourly resolutions, the most recent addition to the Granite Time Series lineup, TTM-R2.1, supports daily and weekly forecasting horizons.

An itemized list of all data sources used to train TTM-R2 and TTM-R2.1 is available at the bottom of the TTM-R2/R2.1 Hugging Face model card. A full list of variants can be found within the “Files and versions” tab.

# Top performance in a tiny package
On Salesforce’s GIFT-Eval Time Series Forecasting Leaderboard, a comprehensive benchmark evaluating time series model performance on multivariate inputs across 24 datasets that span 7 domains, 10 frequencies, and prediction lengths ranging from short to long-term forecasts, TTM-R2 models (including the new TTM-R2.1 variants) top all models for point forecasting accuracy as measured by mean absolute scaled error (MASE).3 TTM-R2 also ranks in the top 5 for probabilistic forecasting, as measured by continuous ranked probability score (CRPS).

It's worth noting that TTM models achieve these rankings by outperforming models many times their size. At “tiny” sizes of 1–5M parameters, TTM models are hundreds of times smaller than the 2nd and 3rd place models by MASE, Google’s TimesFM-2.0 (500M parameters) and Amazon’s Chronos-Bolt-Base (205M parameters).

Increased versatility for forecasting use cases
The TTM-R2.1 release includes an assortment of models with varying context lengths and forecasting horizons. Whereas the previously TTM-R2 models offer context lengths of 1536, 1024 or 512, TTM-R2.1 includes models with shorter context lengths ranging from 512 to 52, making well-suited to daily and weekly forecasts.

The TTM-R2.1 models do not necessarily supersede their TTM-R2 predecessors. The “best” version of TTM depends on the nature of your data and use case. For instance, Granite-Timeseries-TTM-52-16-ft-R2.1 has a context length of 52 and a prediction length of 16, making it best suited to tasks like analyzing a year’s worth of weekly data points and predicting weekly outcomes over the next few months.

The get_model module simplifies the task selecting the right model variant from the extensive offerings available.

Frequency prefix tuning
The “ ft ” designation included in the names of TTM-R2.1 models indicates “frequency tuning” (or, more formally, frequency prefix tuning). Derived from the prefix tuning techniques used as a lightweight alternative for fine-tuning foundation models for text generation tasks, frequency prefix tuning improves the ability of our time-series foundation models to adjust to variations in your input data.

When enabled, an extra embedding vector—indicating the frequency of your data—is added as a “prefix” to the input of the model alongside information from the context window. As detailed in the TTM technical paper, the model team found that frequency tuning improves performance when pretraining on large collections of datasets with diverse resolutions. During inference, this prefix token allows the model to quickly adapt to the frequency of the input data, which is especially useful when the context length is very short.

Granite Embedding: A new sparse embedding model
Whereas all previous Granite Embedding models (and, furthermore, nearly all embedding models in the modern deep learning era) learn dense embeddings, the newest Granite Embedding model—Granite-Embedding-Sparse-30M-English—has a slightly altered architecture that enables it to learn sparse embeddings.

Optimized for exact matches, keyword search and ranking in English, Granite-Embedding-30M-Sparse balances efficiency and scalability across diverse resource and latency budgets. It's released through Granite Experiments, an IBM Research playground for testing open source ideas to speed up the development cycle.

Why sparse embeddings?
A typical, dense embedding model take a text input (such as a document, sentence or query) and outputs a fixed-size vector embedding. The size of that vector—that is, how many numbers (or dimensions) it contains—is a design choice. Models that learn smaller embeddings are faster, but less precise. Models that learn larger embeddings are slower, but more precise. They’re called “dense” vector embeddings because every dimension stores a specific value.

The individual dimensions of a dense vector embedding don’t directly correspond to attributes of the original input’s semantic meaning in any literal way. Dense vector embeddings are essentially a black box: models can use them to perform useful operations, but we humans can’t interpret them in any meaningful way.

Sparse embeddings are more intuitive. Their embedding size is the same as their vocabulary size: that is, each dimension of the vector embedding corresponds with one of the “words”—or, more accurately, one of the tokens—that the model has learned. The specific value contained in each dimension of a sparse embedding vector reflects the relevance of the word (token) that dimension represents to the input for which the model is generating an embedding. Sparse embeddings are thus quite interpretable.

For shorter text passages, such as tweets, comments or brief product reviews, sparse embeddings can be significantly faster while offering performance better than (or at least equal to) that of dense embeddings. They typically offer strong performance “out of the box” without need for fine-tuning.

Having said that, they’re not without downsides. There’s limited opportunity to improve the performance of a sparse embedding model beyond its original baseline through fine-tuning. For longer text passages, any efficiency advantages begin to fade or even reverse as more and more dimensions are utilized to reflect the relevance of an increasing number of tokens from the model’s vocabulary.

The sparse 30M Granite Embedding model offers performance roughly equivalent to its dense 30M counterpart across information retrieval benchmarks (BEIR) while offering a slight advantage over SPLADE-v3.

# Getting started with Granite 3.2
All Granite 3.2 models are available under the permissive Apache 2.0 license on Hugging Face. Select models are also available on IBM watsonx.ai, as well as through platform partners including (in alphabetical order) LM Studio, Ollama and Replicate. Moving forward, this article will be updated to reflect expanded platform availability of Granite 3.2 models. 

