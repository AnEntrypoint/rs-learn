use super::{to_prompt_json, Prompt};
use serde_json::Value;

const QE_SYS: &str = "You are an expert at rephrasing questions into queries used in a database retrieval system";
const QA_SYS: &str = "You are Alice and should respond to all questions from the first person perspective of Alice";
const EVAL_SYS: &str = "You are a judge that determines if answers to questions match a gold standard answer";
const EVAL_ADD_SYS: &str = "You are a judge that determines whether a baseline graph building result from a list of messages is better than a candidate graph building result based on the same messages.";

pub fn query_expansion(query: &Value) -> Prompt {
    let user = format!(
        "Bob is asking Alice a question, are you able to rephrase the question into a simpler one about Alice in the third person that maintains the relevant context?
<QUESTION>
{q}
</QUESTION>

Return JSON: {{\"query\":\"string\"}}
",
        q = to_prompt_json(query),
    );
    Prompt { system: QE_SYS.into(), user, schema: "query" }
}

pub struct QaCtx<'a> {
    pub entity_summaries: &'a Value,
    pub facts: &'a Value,
    pub query: &'a str,
}

pub fn qa_prompt(ctx: &QaCtx) -> Prompt {
    let user = format!(
        "Your task is to briefly answer the question in the way that you think Alice would answer the question.
You are given the following entity summaries and facts to help you determine the answer to your question.
<ENTITY_SUMMARIES>
{ents}
</ENTITY_SUMMARIES>
<FACTS>
{facts}
</FACTS>
<QUESTION>
{q}
</QUESTION>

Return JSON: {{\"ANSWER\":\"string\"}}
",
        ents = to_prompt_json(ctx.entity_summaries),
        facts = to_prompt_json(ctx.facts),
        q = ctx.query,
    );
    Prompt { system: QA_SYS.into(), user, schema: "ANSWER" }
}

pub struct EvalCtx<'a> {
    pub query: &'a str,
    pub answer: &'a str,
    pub response: &'a str,
}

pub fn eval_prompt(ctx: &EvalCtx) -> Prompt {
    let user = format!(
        "Given the QUESTION and the gold standard ANSWER determine if the RESPONSE to the question is correct or incorrect. Although the RESPONSE may be more verbose, mark it as correct as long as it references the same topic as the gold standard ANSWER. Also include your reasoning for the grade.
<QUESTION>
{q}
</QUESTION>
<ANSWER>
{a}
</ANSWER>
<RESPONSE>
{r}
</RESPONSE>

Return JSON: {{\"is_correct\":false,\"reasoning\":\"string\"}}
",
        q = ctx.query,
        a = ctx.answer,
        r = ctx.response,
    );
    Prompt { system: EVAL_SYS.into(), user, schema: "eval" }
}

pub struct EvalAddCtx<'a> {
    pub previous_messages: &'a str,
    pub message: &'a str,
    pub baseline: &'a str,
    pub candidate: &'a str,
}

pub fn eval_add_episode_results(ctx: &EvalAddCtx) -> Prompt {
    let user = format!(
        "Given the following PREVIOUS MESSAGES and MESSAGE, determine if the BASELINE graph data extracted from the conversation is higher quality than the CANDIDATE graph data extracted from the conversation.

Return False if the BASELINE extraction is better, and True otherwise. If the CANDIDATE extraction and BASELINE extraction are nearly identical in quality, return True. Add your reasoning for your decision to the reasoning field.

<PREVIOUS MESSAGES>
{prev}
</PREVIOUS MESSAGES>
<MESSAGE>
{msg}
</MESSAGE>

<BASELINE>
{baseline}
</BASELINE>

<CANDIDATE>
{candidate}
</CANDIDATE>

Return JSON: {{\"candidate_is_worse\":false,\"reasoning\":\"string\"}}
",
        prev = ctx.previous_messages,
        msg = ctx.message,
        baseline = ctx.baseline,
        candidate = ctx.candidate,
    );
    Prompt { system: EVAL_ADD_SYS.into(), user, schema: "eval_add_episode" }
}
