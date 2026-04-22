use super::snippets::{MAX_SUMMARY_CHARS, SUMMARY_INSTRUCTIONS};
use super::{to_prompt_json, Prompt};
use serde_json::Value;

const PAIR_SYS: &str = "You are a helpful assistant that combines summaries into a single dense factual summary.";
const CTX_SYS: &str = "You are a helpful assistant that generates detailed, information-dense summaries and attributes from provided text.";
const DESC_SYS: &str = "You are a helpful assistant that describes provided contents in a single sentence.";

pub fn summarize_pair(node_summaries: &Value) -> Prompt {
    let user = format!(
        "Synthesize the information from the following two summaries into a single information-dense summary.

IMPORTANT:
- Preserve all materially relevant names, roles, places, dates, counts, and changes over time that are explicitly supported.
- Prefer compact factual sentences over vague thematic phrasing.
- SUMMARIES MUST BE LESS THAN {max} CHARACTERS.

Summaries:
{s}

Return JSON: {{\"summary\":\"string\"}}
",
        max = MAX_SUMMARY_CHARS,
        s = to_prompt_json(node_summaries),
    );
    Prompt { system: PAIR_SYS.into(), user, schema: "summary" }
}

pub struct ContextCtx<'a> {
    pub previous_episodes: &'a Value,
    pub episode_content: &'a Value,
    pub node_name: &'a str,
    pub node_summary: &'a str,
    pub attributes: &'a Value,
}

pub fn summarize_context(ctx: &ContextCtx) -> Prompt {
    let user = format!(
        "Given the MESSAGES and the ENTITY name, create a summary for the ENTITY. Your summary must only use information from the provided MESSAGES. Your summary should also only contain information relevant to the provided ENTITY.

In addition, extract any values for the provided entity properties based on their descriptions.
If the value of the entity property cannot be found in the current context, set the value of the property to null.

{sum_instr}

<MESSAGES>
{prev}
{cur}
</MESSAGES>

<ENTITY>
{name}
</ENTITY>

<ENTITY CONTEXT>
{summary}
</ENTITY CONTEXT>

<ATTRIBUTES>
{attrs}
</ATTRIBUTES>

Return JSON: {{\"summary\":\"string\",\"attributes\":{{}}}}
",
        sum_instr = SUMMARY_INSTRUCTIONS,
        prev = to_prompt_json(ctx.previous_episodes),
        cur = to_prompt_json(ctx.episode_content),
        name = ctx.node_name,
        summary = ctx.node_summary,
        attrs = to_prompt_json(ctx.attributes),
    );
    Prompt { system: CTX_SYS.into(), user, schema: "summary_with_attributes" }
}

pub fn summary_description(summary: &Value) -> Prompt {
    let user = format!(
        "Create a short one sentence description of the summary that explains what kind of information is summarized.
Summaries must be under {max} characters.

Summary:
{s}

Return JSON: {{\"description\":\"string\"}}
",
        max = MAX_SUMMARY_CHARS,
        s = to_prompt_json(summary),
    );
    Prompt { system: DESC_SYS.into(), user, schema: "description" }
}
