use super::snippets::{MAX_SUMMARY_CHARS, SUMMARY_INSTRUCTIONS};
use super::{to_prompt_json, Prompt};
use serde_json::Value;

const MSG_SYS: &str = "You are an entity extraction specialist for conversational messages. NEVER extract abstract concepts, feelings, or generic words.";
const JSON_SYS: &str = "You are an entity extraction specialist for JSON data. NEVER extract abstract concepts, dates, or generic field values.";
const TEXT_SYS: &str = "You are an entity extraction specialist for unstructured text. NEVER extract abstract concepts, feelings, or generic words.";
const CLASSIFY_SYS: &str = "You are an entity classification specialist. NEVER assign types not listed in ENTITY TYPES.";
const ATTRS_SYS: &str = "You are an entity attribute extraction specialist. NEVER hallucinate or infer values not explicitly stated.";
const SUMMARY_SYS: &str = "You are a helpful assistant that extracts entity summaries from the provided text.";
const BATCH_SUMMARY_SYS: &str = "You are a helpful assistant that generates concise entity summaries from provided context.";

pub struct ExtractCtx<'a> {
    pub entity_types: &'a str,
    pub previous_episodes: &'a Value,
    pub episode_content: &'a str,
    pub custom_extraction_instructions: Option<&'a str>,
    pub source_description: Option<&'a str>,
}

pub fn extract_message(ctx: &ExtractCtx) -> Prompt {
    let custom = ctx.custom_extraction_instructions.unwrap_or("");
    let user = format!(
        "
NEVER extract any of the following:
- Pronouns (you, me, I, he, she, they, we, us, it, them, him, her, this, that, those)
- Abstract concepts or feelings (joy, balance, growth, resilience, happiness, passion, motivation)
- Generic common nouns or bare object words (day, life, people, work, stuff, things, food, time,
  way, tickets, supplies, clothes, keys, gear)
- Generic media/content nouns unless uniquely identified in the node name itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the node name itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on sentence context rather than the node name itself
- Sentence fragments or clauses
- Adjectives or descriptive phrases
- Duplicate references to the same real-world entity. Extract each entity at most once per message.
- Bare relational or kinship terms (dad, mom, sister, brother, spouse, friend, boss, pet, dog, cat)
  unless qualified with a possessor (e.g., \"Nisha's dad\" is acceptable, \"dad\" alone is not).
- Bare generic objects that cannot be meaningfully qualified.

Your task is to extract entity nodes that are EXPLICITLY mentioned in the CURRENT MESSAGE.
Pronoun references such as he/she/they should be disambiguated to the names of the reference entities.
Only extract distinct entities from the CURRENT MESSAGE.

<ENTITY TYPES>
{entity_types}
</ENTITY TYPES>

<PREVIOUS MESSAGES>
{prev}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{cur}
</CURRENT MESSAGE>

1. Speaker Extraction: Always extract the speaker (the part before the colon : in each dialogue line) as the first entity node.
2. Entity Identification: Extract named entities and specific, concrete things explicitly mentioned. Only extract entities specific enough to be uniquely identifiable.
3. Entity Classification: Use the descriptions in ENTITY TYPES to classify each extracted entity. Assign the appropriate entity_type_id for each one.
4. Exclusions: Do NOT extract entities representing relationships, actions, dates, or times.
5. Specificity: Always use the most specific form mentioned.
6. Formatting: Be explicit and unambiguous in naming entities.

{custom}

Respond with ONLY this JSON shape: {{\"extracted_entities\":[{{\"name\":\"string\",\"entity_type_id\":0}}]}}
",
        entity_types = ctx.entity_types,
        prev = to_prompt_json(ctx.previous_episodes),
        cur = ctx.episode_content,
        custom = custom,
    );
    Prompt { system: MSG_SYS.into(), user, schema: "extracted_entities" }
}

pub fn extract_json(ctx: &ExtractCtx) -> Prompt {
    let custom = ctx.custom_extraction_instructions.unwrap_or("");
    let source = ctx.source_description.unwrap_or("");
    let user = format!(
        "
NEVER extract:
- Date, time, or timestamp values
- Abstract concepts or generic field values (e.g., true, active, pending)
- Numeric IDs or codes that are not meaningful entity names
- Bare relational or kinship terms unless qualified with a possessor name
- Bare generic objects or common nouns unless qualified with a distinguishing detail
- Generic media/content nouns unless uniquely identified
- Generic event/activity nouns unless uniquely identified
- Broad institutional nouns unless explicitly named or uniquely qualified

Extract entities from the JSON and classify each using the ENTITY TYPES above.

<ENTITY TYPES>
{entity_types}
</ENTITY TYPES>

<SOURCE DESCRIPTION>
{source}
</SOURCE DESCRIPTION>

<JSON>
{cur}
</JSON>

Guidelines:
1. Extract the primary entity the JSON represents.
2. Extract named entities referenced in other properties.
3. Only extract entities specific enough to be uniquely identifiable.
4. Be explicit in naming entities.
5. Use the most specific form present.

{custom}

Respond with ONLY this JSON shape: {{\"extracted_entities\":[{{\"name\":\"string\",\"entity_type_id\":0}}]}}
",
        entity_types = ctx.entity_types,
        source = source,
        cur = ctx.episode_content,
        custom = custom,
    );
    Prompt { system: JSON_SYS.into(), user, schema: "extracted_entities" }
}

pub fn extract_text(ctx: &ExtractCtx) -> Prompt {
    let custom = ctx.custom_extraction_instructions.unwrap_or("");
    let user = format!(
        "
NEVER extract:
- Pronouns (you, me, he, she, they, it, them, him, her, we, us, this, that, those)
- Abstract concepts (joy, balance, growth, resilience, passion, motivation)
- Generic common nouns or bare object words
- Generic media/content nouns unless uniquely identified
- Generic event/activity nouns unless uniquely identified
- Broad institutional nouns unless explicitly named or uniquely qualified
- Sentence fragments or clauses as entity names
- Bare relational or kinship terms unless qualified with a possessor
- Bare generic objects that cannot be meaningfully qualified

Extract entities from the TEXT that are explicitly mentioned.
For each entity, classify it using the ENTITY TYPES above.
Only extract entities specific enough to be uniquely identifiable.

<ENTITY TYPES>
{entity_types}
</ENTITY TYPES>

<TEXT>
{cur}
</TEXT>

{custom}

Respond with ONLY this JSON shape: {{\"extracted_entities\":[{{\"name\":\"string\",\"entity_type_id\":0}}]}}
",
        entity_types = ctx.entity_types,
        cur = ctx.episode_content,
        custom = custom,
    );
    Prompt { system: TEXT_SYS.into(), user, schema: "extracted_entities" }
}

pub struct ClassifyCtx<'a> {
    pub previous_episodes: &'a Value,
    pub episode_content: &'a str,
    pub extracted_entities: &'a str,
    pub entity_types: &'a str,
}

pub fn classify_nodes(ctx: &ClassifyCtx) -> Prompt {
    let user = format!(
        "
<PREVIOUS MESSAGES>
{prev}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{cur}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
{ents}
</EXTRACTED ENTITIES>

<ENTITY TYPES>
{etypes}
</ENTITY TYPES>

Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted entities.

Guidelines:
1. Each entity must have exactly one type.
2. NEVER use types not listed in ENTITY TYPES.
3. If none of the provided entity types accurately classify an extracted entity, the type should be set to None.

Return JSON: {{\"entity_classifications\":[{{\"name\":\"string\",\"entity_type\":\"string|null\"}}]}}
",
        prev = to_prompt_json(ctx.previous_episodes),
        cur = ctx.episode_content,
        ents = ctx.extracted_entities,
        etypes = ctx.entity_types,
    );
    Prompt { system: CLASSIFY_SYS.into(), user, schema: "entity_classifications" }
}

pub struct NodeAttrsCtx<'a> {
    pub previous_episodes: &'a Value,
    pub episode_content: &'a Value,
    pub node: &'a Value,
}

pub fn extract_attributes(ctx: &NodeAttrsCtx) -> Prompt {
    let user = format!(
        "
Given the MESSAGES and the following ENTITY, update any of its attributes based on the information provided in MESSAGES. Use the provided attribute descriptions to better understand how each attribute should be determined.

Guidelines:
1. NEVER hallucinate or infer property values — only use values explicitly stated in the MESSAGES.
2. Only use the provided MESSAGES and ENTITY to set attribute values.

<MESSAGES>
{prev}
{cur}
</MESSAGES>

<ENTITY>
{node}
</ENTITY>

Return JSON: {{\"attributes\":{{}}}}
",
        prev = to_prompt_json(ctx.previous_episodes),
        cur = to_prompt_json(ctx.episode_content),
        node = to_prompt_json(ctx.node),
    );
    Prompt { system: ATTRS_SYS.into(), user, schema: "attributes" }
}

pub struct SummaryCtx<'a> {
    pub previous_episodes: &'a Value,
    pub episode_content: &'a Value,
    pub node: Option<&'a Value>,
    pub entities: Option<&'a Value>,
}

pub fn extract_summary(ctx: &SummaryCtx) -> Prompt {
    let node = ctx.node.cloned().unwrap_or(Value::Null);
    let user = format!(
        "
Given the MESSAGES and the ENTITY, update the summary that combines relevant information about the entity from the messages and relevant information from the existing summary. Summary must be under {max} characters.

{sum_instr}

<MESSAGES>
{prev}
{cur}
</MESSAGES>

<ENTITY>
{node}
</ENTITY>

Return JSON: {{\"summary\":\"string\"}}
",
        max = MAX_SUMMARY_CHARS,
        sum_instr = SUMMARY_INSTRUCTIONS,
        prev = to_prompt_json(ctx.previous_episodes),
        cur = to_prompt_json(ctx.episode_content),
        node = to_prompt_json(&node),
    );
    Prompt { system: SUMMARY_SYS.into(), user, schema: "summary" }
}

pub fn extract_summaries_batch(ctx: &SummaryCtx) -> Prompt {
    let entities = ctx.entities.cloned().unwrap_or(Value::Array(vec![]));
    let user = format!(
        "
Given the MESSAGES and a list of ENTITIES, generate an updated summary for each entity that needs one.
Each summary must be under {max} characters.

{sum_instr}

<MESSAGES>
{prev}
{cur}
</MESSAGES>

<ENTITIES>
{ents}
</ENTITIES>

For each entity, combine relevant information from the MESSAGES with any existing summary content.
Only return summaries for entities that have meaningful information to summarize.

Return JSON: {{\"summaries\":[{{\"name\":\"string\",\"summary\":\"string\"}}]}}
",
        max = MAX_SUMMARY_CHARS,
        sum_instr = SUMMARY_INSTRUCTIONS,
        prev = to_prompt_json(ctx.previous_episodes),
        cur = to_prompt_json(ctx.episode_content),
        ents = to_prompt_json(&entities),
    );
    Prompt { system: BATCH_SUMMARY_SYS.into(), user, schema: "summaries" }
}

const EPISODE_SUMMARY_SYS: &str = "You maintain detailed, information-dense entity memories from episode text. Use ONLY facts explicitly stated in EPISODES and durable facts already present in EXISTING_SUMMARY. NEVER infer beyond what is directly supported. Write 2-6 dense sentences in third person. Return only the summary text.";

pub fn extract_entity_summaries_from_episodes(ctx: &SummaryCtx) -> Prompt {
    let entities = ctx.entities.cloned().unwrap_or(Value::Array(vec![]));
    let user = format!(
        "NEVER include meta-language about the summarization process. Use ONLY facts from the provided EPISODES.
Each summary must be under {max} characters. Write 2-6 dense sentences in third person.

For each entity below, generate an updated summary using ONLY the provided EPISODES and any existing summary already on the entity.

<EPISODES>
{prev}
{cur}
</EPISODES>

<ENTITIES>
{ents}
</ENTITIES>

Only return summaries for entities that have meaningful information to summarize.

Return JSON: {{\"summaries\":[{{\"name\":\"string\",\"summary\":\"string\"}}]}}
",
        max = MAX_SUMMARY_CHARS,
        prev = to_prompt_json(ctx.previous_episodes),
        cur = to_prompt_json(ctx.episode_content),
        ents = to_prompt_json(&entities),
    );
    Prompt { system: EPISODE_SUMMARY_SYS.into(), user, schema: "summaries" }
}
