# Style Guidelines
## Abstract
150-300 words MAX.
Default structure: (1) Problem/Gap, (2) Approach, (3) Key Results (with specific metrics), (4) Main Implication.
Be specific and verifiable.
CITATIONS ARE STRICTLY FORBIDDEN. Do NOT include ANY citations in the Abstract.
Do NOT use generic phrases like "In this paper, we propose...". Jump straight into the problem or approach.

**Intuitive and Accessible Vocabulary:**
    Write in a clear, straightforward, and professional style. Prioritize clarity for non-native speakers. Replace robotic, overly academic, or technical jargon with simple, intuitive everyday terms. 
    - **DON'T:** "Crucially, the system leverages vector embeddings to facilitate contextual alignment."
    - **DO:** "The system uses vector embeddings to match the text to the context."
    - *Forbidden Verbs:* leveraging, situating, ensuring, utilizing, facilitating. Use simple equivalents (uses, places, makes sure, helps).
    - *Forbidden Adverbs:* crucially, strictly, importantly, topically, additionally. 

**Dynamic Sentence Structure:**
    Vary your sentence lengths and structures. Do not start consecutive sentences the same way. Keep sentences distinct and easy to read. Use concise, direct one-liners where appropriate to punch up readability.
    - **DON'T:** "The system first loads the data. The system then processes the data. The system finally saves the data."
    - **DO:** "The process begins by loading the user's data. Once loaded, the system processes it to extract metadata. Finally, the results are saved to the disk."

## Introduction
Open with the problem and its concrete impact.
Identify what's missing in current solutions using evidence.
State your contribution as specific, falsifiable claims.
End with brief paper roadmap.
Justify claims with evidence, don't just assert.

Scientific Writing & Tone Guidelines
1. **Logical Derivation Over Empirical Claims:** 
   Never make subjective, absolute, or empirical claims about performance, output quality, or user behavior unless you have a specific citation to back it up. Instead, describe the *logical, mechanical effects* of the system's design or algorithms.
   - **DON'T:** "This method is highly effective and improves accuracy."
   - **DO:** "This method provides an explicit filtering mechanism, preventing the algorithm from processing out-of-bounds data."

2. **No Baseless Statements:**
   Every technical statement must either be logically derived from the provided context or explicitly backed by a provided citation. Ungrounded claims in academic texts are strictly prohibited. 
   - **DON'T:** "Studies show that local models are weaker than cloud models."
   - **DO:** "Because local models operate with smaller parameter counts and quantized weights, they possess less reasoning capacity than uncompressed cloud models [CITE]."

3. **Intuitive and Accessible Vocabulary:**
   Write in a clear, straightforward, and professional style. Prioritize clarity for non-native speakers. Replace robotic, overly academic, or technical jargon with simple, intuitive everyday terms. 
   - **DON'T:** "Crucially, the system leverages vector embeddings to facilitate contextual alignment."
   - **DO:** "The system uses vector embeddings to match the text to the context."
   - *Forbidden Verbs:* leveraging, situating, ensuring, utilizing, facilitating. Use simple equivalents (uses, places, makes sure, helps).
   - *Forbidden Adverbs:* crucially, strictly, importantly, topically, additionally. 

4. **Dynamic Sentence Structure:**
   Vary your sentence lengths and structures. Do not start consecutive sentences the same way. Keep sentences distinct and easy to read. Use concise, direct one-liners where appropriate to punch up readability.
   - **DON'T:** "The system first loads the data. The system then processes the data. The system finally saves the data."
   - **DO:** "The process begins by loading the user's data. Once loaded, the system processes it to extract metadata. Finally, the results are saved to the disk."

## Related Work
Group by approach/theme, not chronologically. For each group:
- What they did (method + reported results)
- Limitations relative to this work
- Direct comparison where applicable
Avoid generic praise. Be precise about differences. Cite liberally.

Scientific Writing & Tone Guidelines
1. **Logical Derivation Over Empirical Claims:** 
   Never make subjective, absolute, or empirical claims about performance, output quality, or user behavior unless you have a specific citation to back it up. Instead, describe the *logical, mechanical effects* of the system's design or algorithms.
   - **DON'T:** "This method is highly effective and improves accuracy."
   - **DO:** "This method provides an explicit filtering mechanism, preventing the algorithm from processing out-of-bounds data."

2. **No Baseless Statements:**
   Every technical statement must either be logically derived from the provided context or explicitly backed by a provided citation. Ungrounded claims in academic texts are strictly prohibited. 
   - **DON'T:** "Studies show that local models are weaker than cloud models."
   - **DO:** "Because local models operate with smaller parameter counts and quantized weights, they possess less reasoning capacity than uncompressed cloud models [CITE]."

3. **Intuitive and Accessible Vocabulary:**
   Write in a clear, straightforward, and professional style. Prioritize clarity for non-native speakers. Replace robotic, overly academic, or technical jargon with simple, intuitive everyday terms. 
   - **DON'T:** "Crucially, the system leverages vector embeddings to facilitate contextual alignment."
   - **DO:** "The system uses vector embeddings to match the text to the context."
   - *Forbidden Verbs:* leveraging, situating, ensuring, utilizing, facilitating. Use simple equivalents (uses, places, makes sure, helps).
   - *Forbidden Adverbs:* crucially, strictly, importantly, topically, additionally. 

4. **Dynamic Sentence Structure:**
   Vary your sentence lengths and structures. Do not start consecutive sentences the same way. Keep sentences distinct and easy to read. Use concise, direct one-liners where appropriate to punch up readability.
   - **DON'T:** "The system first loads the data. The system then processes the data. The system finally saves the data."
   - **DO:** "The process begins by loading the user's data. Once loaded, the system processes it to extract metadata. Finally, the results are saved to the disk."

## Methods
Reproducibility is the goal. If possible and relevant, include:
- Architecture/algorithm with justification for key choices
- Hyperparameters, dataset details, compute resources
- Baseline comparisons (what and why)
- Evaluation metrics with rationale
Use present tense. Avoid implementation details unless critical.

Scientific Writing & Tone Guidelines
1. **Logical Derivation Over Empirical Claims:** 
   Never make subjective, absolute, or empirical claims about performance, output quality, or user behavior unless you have a specific citation to back it up. Instead, describe the *logical, mechanical effects* of the system's design or algorithms.
   - **DON'T:** "This method is highly effective and improves accuracy."
   - **DO:** "This method provides an explicit filtering mechanism, preventing the algorithm from processing out-of-bounds data."

2. **No Baseless Statements:**
   Every technical statement must either be logically derived from the provided context or explicitly backed by a provided citation. Ungrounded claims in academic texts are strictly prohibited. 
   - **DON'T:** "Studies show that local models are weaker than cloud models."
   - **DO:** "Because local models operate with smaller parameter counts and quantized weights, they possess less reasoning capacity than uncompressed cloud models [CITE]."

3. **Intuitive and Accessible Vocabulary:**
   Write in a clear, straightforward, and professional style. Prioritize clarity for non-native speakers. Replace robotic, overly academic, or technical jargon with simple, intuitive everyday terms. 
   - **DON'T:** "Crucially, the system leverages vector embeddings to facilitate contextual alignment."
   - **DO:** "The system uses vector embeddings to match the text to the context."
   - *Forbidden Verbs:* leveraging, situating, ensuring, utilizing, facilitating. Use simple equivalents (uses, places, makes sure, helps).
   - *Forbidden Adverbs:* crucially, strictly, importantly, topically, additionally. 

4. **Dynamic Sentence Structure:**
   Vary your sentence lengths and structures. Do not start consecutive sentences the same way. Keep sentences distinct and easy to read. Use concise, direct one-liners where appropriate to punch up readability.
   - **DON'T:** "The system first loads the data. The system then processes the data. The system finally saves the data."
   - **DO:** "The process begins by loading the user's data. Once loaded, the system processes it to extract metadata. Finally, the results are saved to the disk."

## Results
Present experiment outcomes with relevant metrics or observations.
Compare results against expected improvements or baselines if available.
Never fabricate data or results.

Scientific Writing & Tone Guidelines
1. **Logical Derivation Over Empirical Claims:** 
   Never make subjective, absolute, or empirical claims about performance, output quality, or user behavior unless you have a specific citation to back it up. Instead, describe the *logical, mechanical effects* of the system's design or algorithms.
   - **DON'T:** "This method is highly effective and improves accuracy."
   - **DO:** "This method provides an explicit filtering mechanism, preventing the algorithm from processing out-of-bounds data."

2. **No Baseless Statements:**
   Every technical statement must either be logically derived from the provided context or explicitly backed by a provided citation. Ungrounded claims in academic texts are strictly prohibited. 
   - **DON'T:** "Studies show that local models are weaker than cloud models."
   - **DO:** "Because local models operate with smaller parameter counts and quantized weights, they possess less reasoning capacity than uncompressed cloud models [CITE]."

3. **Intuitive and Accessible Vocabulary:**
   Write in a clear, straightforward, and professional style. Prioritize clarity for non-native speakers. Replace robotic, overly academic, or technical jargon with simple, intuitive everyday terms. 
   - **DON'T:** "Crucially, the system leverages vector embeddings to facilitate contextual alignment."
   - **DO:** "The system uses vector embeddings to match the text to the context."
   - *Forbidden Verbs:* leveraging, situating, ensuring, utilizing, facilitating. Use simple equivalents (uses, places, makes sure, helps).
   - *Forbidden Adverbs:* crucially, strictly, importantly, topically, additionally. 

4. **Dynamic Sentence Structure:**
   Vary your sentence lengths and structures. Do not start consecutive sentences the same way. Keep sentences distinct and easy to read. Use concise, direct one-liners where appropriate to punch up readability.
   - **DON'T:** "The system first loads the data. The system then processes the data. The system finally saves the data."
   - **DO:** "The process begins by loading the user's data. Once loaded, the system processes it to extract metadata. Finally, the results are saved to the disk."

## Discussion
Open by restating main finding in context of hypothesis.
Explain why it worked/failed using specific evidence and results. Acknowledge limitations honestly.
Compare to related work quantitatively where possible.
Speculation allowed but label it clearly.
End with concrete future directions, not vague "explore further.

Scientific Writing & Tone Guidelines
1. **Logical Derivation Over Empirical Claims:** 
   Never make subjective, absolute, or empirical claims about performance, output quality, or user behavior unless you have a specific citation to back it up. Instead, describe the *logical, mechanical effects* of the system's design or algorithms.
   - **DON'T:** "This method is highly effective and improves accuracy."
   - **DO:** "This method provides an explicit filtering mechanism, preventing the algorithm from processing out-of-bounds data."

2. **No Baseless Statements:**
   Every technical statement must either be logically derived from the provided context or explicitly backed by a provided citation. Ungrounded claims in academic texts are strictly prohibited. 
   - **DON'T:** "Studies show that local models are weaker than cloud models."
   - **DO:** "Because local models operate with smaller parameter counts and quantized weights, they possess less reasoning capacity than uncompressed cloud models [CITE]."

3. **Intuitive and Accessible Vocabulary:**
   Write in a clear, straightforward, and professional style. Prioritize clarity for non-native speakers. Replace robotic, overly academic, or technical jargon with simple, intuitive everyday terms. 
   - **DON'T:** "Crucially, the system leverages vector embeddings to facilitate contextual alignment."
   - **DO:** "The system uses vector embeddings to match the text to the context."
   - *Forbidden Verbs:* leveraging, situating, ensuring, utilizing, facilitating. Use simple equivalents (uses, places, makes sure, helps).
   - *Forbidden Adverbs:* crucially, strictly, importantly, topically, additionally. 

4. **Dynamic Sentence Structure:**
   Vary your sentence lengths and structures. Do not start consecutive sentences the same way. Keep sentences distinct and easy to read. Use concise, direct one-liners where appropriate to punch up readability.
   - **DON'T:** "The system first loads the data. The system then processes the data. The system finally saves the data."
   - **DO:** "The process begins by loading the user's data. Once loaded, the system processes it to extract metadata. Finally, the results are saved to the disk."

## Conclusion
Summarize: what you did, what you found (with key metrics), broader implications (realistic, not grandiose), actionable next step(s).
NO new information. Few or no citations.

Scientific Writing & Tone Guidelines
1. **Logical Derivation Over Empirical Claims:** 
   Never make subjective, absolute, or empirical claims about performance, output quality, or user behavior unless you have a specific citation to back it up. Instead, describe the *logical, mechanical effects* of the system's design or algorithms.
   - **DON'T:** "This method is highly effective and improves accuracy."
   - **DO:** "This method provides an explicit filtering mechanism, preventing the algorithm from processing out-of-bounds data."

2. **No Baseless Statements:**
   Every technical statement must either be logically derived from the provided context or explicitly backed by a provided citation. Ungrounded claims in academic texts are strictly prohibited. 
   - **DON'T:** "Studies show that local models are weaker than cloud models."
   - **DO:** "Because local models operate with smaller parameter counts and quantized weights, they possess less reasoning capacity than uncompressed cloud models [CITE]."

3. **Intuitive and Accessible Vocabulary:**
   Write in a clear, straightforward, and professional style. Prioritize clarity for non-native speakers. Replace robotic, overly academic, or technical jargon with simple, intuitive everyday terms. 
   - **DON'T:** "Crucially, the system leverages vector embeddings to facilitate contextual alignment."
   - **DO:** "The system uses vector embeddings to match the text to the context."
   - *Forbidden Verbs:* leveraging, situating, ensuring, utilizing, facilitating. Use simple equivalents (uses, places, makes sure, helps).
   - *Forbidden Adverbs:* crucially, strictly, importantly, topically, additionally. 

4. **Dynamic Sentence Structure:**
   Vary your sentence lengths and structures. Do not start consecutive sentences the same way. Keep sentences distinct and easy to read. Use concise, direct one-liners where appropriate to punch up readability.
   - **DON'T:** "The system first loads the data. The system then processes the data. The system finally saves the data."
   - **DO:** "The process begins by loading the user's data. Once loaded, the system processes it to extract metadata. Finally, the results are saved to the disk."

## Acknowledgements
Format and polish the provided acknowledgements text into a professional academic style.
Keep the original meaning and intent, but ensure proper grammar, flow, and academic tone.
No citations needed. Keep it concise and appropriate for an academic paper.