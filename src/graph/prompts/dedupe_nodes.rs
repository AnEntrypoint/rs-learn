use super::{to_prompt_json, Prompt};
use serde_json::Value;

const SYS: &str = "You are an entity deduplication assistant. NEVER fabricate entity names or mark distinct entities as duplicates.";
const SYS_LIST: &str = "You are an entity deduplication assistant that groups duplicate nodes by UUID.";

pub struct NodeCtx<'a> {
    pub previous_episodes: &'a Value,
    pub episode_content: &'a str,
    pub extracted_node: &'a Value,
    pub entity_type_description: &'a Value,
    pub existing_nodes: &'a Value,
}

pub fn node(ctx: &NodeCtx) -> Prompt {
    let user = format!(r#"
<PREVIOUS MESSAGES>
{prev}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{cur}
</CURRENT MESSAGE>

<NEW ENTITY>
{new}
</NEW ENTITY>

<ENTITY TYPE DESCRIPTION>
{etd}
</ENTITY TYPE DESCRIPTION>

<EXISTING ENTITIES>
{existing}
</EXISTING ENTITIES>

Entities should only be considered duplicates if they refer to the *same real-world object or concept*.
Semantic Equivalence: if a descriptive label in EXISTING ENTITIES clearly refers to a named entity in context, treat them as duplicates.

NEVER mark entities as duplicates if:
- They are related but distinct.
- They have similar names or purposes but refer to separate instances or concepts.

Task:
1. Compare the NEW ENTITY against each EXISTING ENTITY (identified by `candidate_id`).
2. If it refers to the same real-world object or concept, return the `candidate_id` of that match.
3. Return `duplicate_candidate_id = -1` when there is no match or you are unsure.

<EXAMPLE>
NEW ENTITY: "Sam" (Person)
EXISTING ENTITIES: [{{"candidate_id": 0, "name": "Sam", "entity_types": ["Person"], "summary": "Sam enjoys hiking and photography"}}]
Result: duplicate_candidate_id = 0 (same person referenced in conversation)

NEW ENTITY: "NYC"
EXISTING ENTITIES: [{{"candidate_id": 0, "name": "New York City", "entity_types": ["Location"]}}, {{"candidate_id": 1, "name": "New York Knicks", "entity_types": ["Organization"]}}]
Result: duplicate_candidate_id = 0 (same location, abbreviated name)

NEW ENTITY: "Java" (programming language)
EXISTING ENTITIES: [{{"candidate_id": 0, "name": "Java", "entity_types": ["Location"], "summary": "An island in Indonesia"}}]
Result: duplicate_candidate_id = -1 (same name but distinct real-world things)

NEW ENTITY: "Marco's car"
EXISTING ENTITIES: [{{"candidate_id": 0, "name": "Marco's vehicle", "entity_types": ["Entity"], "summary": "Marco drives a red sedan."}}]
Result: duplicate_candidate_id = 0 (synonym — "car" and "vehicle" refer to the same thing, same possessor)
</EXAMPLE>

Return JSON: {{"id":0,"name":"string","duplicate_candidate_id":-1}}
"#,
        prev = to_prompt_json(ctx.previous_episodes),
        cur = ctx.episode_content,
        new = to_prompt_json(ctx.extracted_node),
        etd = to_prompt_json(ctx.entity_type_description),
        existing = to_prompt_json(ctx.existing_nodes),
    );
    Prompt { system: SYS.into(), user, schema: "node_duplicate" }
}

pub struct NodesCtx<'a> {
    pub previous_episodes: &'a Value,
    pub episode_content: &'a str,
    pub extracted_nodes: &'a Value,
    pub existing_nodes: &'a Value,
}

pub fn nodes(ctx: &NodesCtx) -> Prompt {
    let n = ctx.extracted_nodes.as_array().map(|a| a.len()).unwrap_or(0);
    let n1 = if n == 0 { 0 } else { n - 1 };
    let user = format!(r#"
<PREVIOUS MESSAGES>
{prev}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{cur}
</CURRENT MESSAGE>

<ENTITIES>
{ents}
</ENTITIES>

<EXISTING ENTITIES>
{existing}
</EXISTING ENTITIES>

Each of the above ENTITIES was extracted from the CURRENT MESSAGE.
For each entity, determine if it is a duplicate of any EXISTING ENTITY.
Entities should only be considered duplicates if they refer to the *same real-world object or concept*.

NEVER mark entities as duplicates if:
- They are related but distinct.
- They have similar names or purposes but refer to separate instances or concepts.

Task:
ENTITIES contains {n} entities with IDs 0 through {n1}.
Your response MUST include EXACTLY {n} resolutions with IDs 0 through {n1}. Do not skip or add IDs.

For every entity, provide:
- `id`: integer id from ENTITIES
- `name`: the best full name for the entity (preserve the original name unless a duplicate has a more complete name)
- `duplicate_candidate_id`: the `candidate_id` of the EXISTING ENTITY that is the best duplicate match, or -1 if there is no duplicate

Return JSON: {{"entity_resolutions":[{{"id":0,"name":"string","duplicate_candidate_id":-1}}]}}
"#,
        prev = to_prompt_json(ctx.previous_episodes),
        cur = ctx.episode_content,
        ents = to_prompt_json(ctx.extracted_nodes),
        existing = to_prompt_json(ctx.existing_nodes),
    );
    Prompt { system: SYS.into(), user, schema: "entity_resolutions" }
}

pub fn node_list(nodes: &Value) -> Prompt {
    let user = format!(r#"
Given the following context, deduplicate a list of nodes:

<NODES>
{nodes}
</NODES>

Task:
1. Group nodes together such that all duplicate nodes are in the same list of uuids.
2. All duplicate uuids should be grouped together in the same list.
3. Also return a new summary that synthesizes the summaries into a new short summary.

Guidelines:
1. Each uuid from the list of nodes should appear EXACTLY once in your response.
2. If a node has no duplicates, it should appear in the response in a list of only one uuid.

Return JSON: {{"groups":[{{"uuids":["..."],"summary":"string"}}]}}
"#,
        nodes = to_prompt_json(nodes),
    );
    Prompt { system: SYS_LIST.into(), user, schema: "groups" }
}
