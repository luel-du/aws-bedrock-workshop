# Building with Amazon Bedrock: Workshop Notes

Hands-on notes from the AWS "Building with Amazon Bedrock" workshop. Every lab follows the same shape: a small
Python library that calls Bedrock through `boto3`, and a Streamlit page on top of it. The labs ran in the
workshop's browser-based VS Code environment in `us-west-2`.

## Labs completed

| Category | Labs | What was built |
| --- | --- | --- |
| Text patterns | I-1, I-2, I-3, I-4 | RAG chatbot, document summarization, response streaming, embeddings search |
| Security and safeguards | S-1, S-2, S-3, S-4, S-5, S-6 | Guardrails from the console and from code, invoked with trace; content, PII, and prompt-attack policies |
| Bonus | O-1, O-2 | Prompt template showcase, text generation playground with inference parameters |

Screenshots are in `screenshots/`, named by lab.

## Key concepts

- **One API, many models.** `bedrock-runtime` exposes `invoke_model` (model-specific JSON body) and `converse`
  (one message format for every chat model). Model IDs such as `us.amazon.nova-lite-v1:0` are cross-region
  inference profiles, so the same ID works from several regions.
- **Embeddings turn text into vectors.** A Titan embeddings model maps each document and each question to a
  vector; nearest vectors are the most relevant documents. This is the retrieval half of RAG.
- **RAG = retrieve, then generate.** Retrieved passages are placed in the prompt so the model answers from
  your data instead of its training data.
- **Streaming** returns the response token by token through `invoke_model_with_response_stream` or
  `converse_stream`, so the UI can render while the model is still writing.
- **Guardrails are a separate resource.** A guardrail holds policies (denied topics, content filters, PII
  handling, prompt-attack detection, word filters). It is attached at invoke time by ID and version, applies to
  input and output, and returns a trace explaining what it did.
- **Inference parameters** (temperature, top P, max tokens) shape randomness and length and are passed on every call.
- **Prompt templates** separate the reusable instruction from the user's input, which is what the showcase
  and summarization labs do.

## Common architecture

```mermaid
flowchart LR
    U[User in browser] --> ST[Streamlit app<br/>*_app.py]
    ST --> LIB[Python library<br/>*_lib.py]
    LIB --> RT[boto3 bedrock-runtime]
    RT --> FM[Foundation model<br/>Nova Lite / Claude / Titan]
    G[Bedrock Guardrail<br/>id + version] -. applied on input and output .-> RT
```

Every app was started the same way from the lab folder:

```bash
streamlit run <lab>_app.py --server.port 8080
```

## Text patterns

### Lab I-1: Chatbot with RAG

A chat interface whose answers are grounded in a document set about Amazon Bedrock. Each turn, the question is
embedded, the closest passages are retrieved, and the model answers from those passages while keeping the chat
history. The bot's reply to a vague "what can you do with it?" shows the grounding at work: it notes that its
sources only cover Bedrock, then answers from them.

```mermaid
flowchart LR
    Q[User question] --> E[Embed question<br/>Titan Embeddings]
    E --> VS[(Vector index<br/>Bedrock docs)]
    VS --> C[Top passages]
    C --> P[Prompt = history + passages + question]
    P --> M[Chat model via converse]
    M --> A[Answer]
```

Steps:
1. Load and chunk the source document, embed every chunk once, and keep the vectors in an in-memory index.
2. On each message, embed the question and select the nearest chunks.
3. Call the chat model with the conversation history plus the retrieved context.
4. Keep the history in `st.session_state` so follow-up questions work.

Minimal sketch of the pattern:

```python
import boto3

runtime = boto3.client("bedrock-runtime", region_name="us-west-2")

def answer(question, history, context_passages):
    system = ("Answer only from the context below. If it is not there, say so.\n\n"
              + "\n\n".join(context_passages))
    messages = history + [{"role": "user", "content": [{"text": question}]}]
    response = runtime.converse(
        modelId="us.amazon.nova-lite-v1:0",
        system=[{"text": system}],
        messages=messages,
        inferenceConfig={"maxTokens": 1024, "temperature": 0.0},
    )
    return response["output"]["message"]["content"][0]["text"]
```

![RAG chatbot answering a question about Bedrock from its sources](screenshots/I-1-rag-chatbot.png)

### Lab I-4: Embeddings search

Semantic search without a generation step. A CSV of Bedrock FAQ entries is embedded once; the question is
embedded at query time and the FAQ rows are ranked by cosine similarity. The screenshot shows the top matches
for "What can you do with Amazon Bedrock?", which are the right FAQ entries even though they do not share the
exact wording.

```mermaid
flowchart LR
    D[FAQ rows] --> ED[Embed once] --> V[(Vectors)]
    Q[Question] --> EQ[Embed] --> S[Cosine similarity vs V]
    V --> S --> R[Ranked rows]
```

Minimal sketch of the pattern:

```python
import json, boto3, numpy as np

runtime = boto3.client("bedrock-runtime", region_name="us-west-2")

def embed(text):
    body = json.dumps({"inputText": text})
    out = runtime.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=body)
    return np.array(json.loads(out["body"].read())["embedding"])

def search(question, rows, row_vectors, k=3):
    q = embed(question)
    scores = row_vectors @ q / (np.linalg.norm(row_vectors, axis=1) * np.linalg.norm(q))
    return [rows[i] for i in np.argsort(-scores)[:k]]
```

![Embeddings search returning the closest FAQ entries](screenshots/I-4-embeddings-search.png)

### Labs I-2 and I-3 in brief

**I-2 Document summarization.** The user supplies the instruction and the app wraps the document in a prompt
template. Prompt used: "summarize the document, highlighting the three most important elements that an intern or
new hire should focus on", run against the Amazon Leadership Principles.

![Document summarization with a custom instruction](screenshots/I-2-document-summarization.png)

**I-3 Response streaming.** The same request through the streaming API, with Streamlit's `st.write_stream`
rendering chunks as they arrive. The Stop button in the corner shows the response is still in flight.

![A story being streamed token by token](screenshots/I-3-response-streaming.png)

## Security and safeguards

### Labs S-1 and S-2: Creating a guardrail (console and code)

The same guardrail was built twice: once by clicking through the console, once with `create_guardrail`. Both
define a denied topic, "Bitcoin", with a natural-language definition and example phrases. The console Test
panel confirms the policy: "I like bitcoin, how about you?" is intercepted and replaced by the configured
blocked message, with no model call made.

```mermaid
flowchart LR
    CG[create_guardrail<br/>topic + content + PII + attack policies] --> GR[Guardrail<br/>DRAFT]
    GR --> CV[create_guardrail_version] --> V1[Version 1]
    V1 -- id + version --> INV[invoke_model / converse]
```

`create_guardrail.py` (abridged; the full file also sets the blocked-input and blocked-output messages):

```python
import boto3, json, random, string

client = boto3.client(service_name="bedrock")

response = client.create_guardrail(
    name="Guardrail-from-Code-" + "".join(random.choices(string.ascii_lowercase, k=8)),
    description="string",
    topicPolicyConfig={
        "topicsConfig": [
            {
                "name": "Bitcoin",
                "definition": "Providing advice, direction, or examples of how to mine, use, or "
                              "interact with Bitcoin, including Cryptocurrency-related third-party services.",
                "examples": [
                    "How do I mine Bitcoin?",
                    "What is the current value of BTC?",
                    "Which instance is the best for crypto mining?",
                    "Is mining cryptocurrency against the terms?",
                    "How do I maximize my Bitcoin profits?",
                ],
                "type": "DENY",
            }
        ]
    },
    blockedInputMessaging="Apologies, this model cannot be used to discuss inappropriate or off-topic content.",
    blockedOutputsMessaging="Apologies, this model cannot be used to discuss inappropriate or off-topic content.",
)
print(response["guardrailId"])
```

![Guardrail created in the console and tested with an off-topic prompt](screenshots/S-1-guardrail-console-test.png)

![Guardrail created from Python, returning its ID](screenshots/S-2-create-guardrail-code.png)

### Lab S-3: Invoking a model with a guardrail and reading the trace

The guardrail ID and version are passed on the invoke call. A harmless prompt ("How do you wash a car?") goes
through unchanged; a harmful one ("How do you hotwire a car?") is stopped, and `trace='ENABLED'` returns the
assessment that explains which filter fired, at what confidence, and what action was taken.

```python
import boto3, json

def invoke_nova_lite_with_guardrails(prompt, guardrail_id, guardrail_version, region="us-west-2"):
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
    model_id = "us.amazon.nova-lite-v1:0"

    request_body = {
        "schemaVersion": "messages-v1",
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 500},
    }
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body),
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        trace="ENABLED",
    )
    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"], result.get("amazon-bedrock-trace")
```

![Allowed prompt answered, blocked prompt returning a MISCONDUCT trace](screenshots/S-3-invoke-with-guardrails-trace.png)

### Labs S-4, S-5, S-6: Policies in action

Each lab adds one policy to a guardrail and tests it from the Streamlit guardrails app, which shows the final
response next to the trace.

| Lab | Policy | Test input | Result in trace |
| --- | --- | --- | --- |
| S-4 Content blocking | Content filter, HATE at HIGH strength | "I hate city people" | `HATE`, confidence HIGH, action BLOCKED |
| S-5 PII masking | Sensitive information policy, NAME and EMAIL set to ANONYMIZE | Customer letter with names and an email | Output rewritten to `{NAME}` and `{EMAIL}`; `modelOutput` in the trace still shows the original text |
| S-6 Prompt attack blocking | Prompt attack filter at HIGH strength | "Pretend to be my grandmother" jailbreak | `PROMPT_ATTACK`, confidence MEDIUM, action BLOCKED |

Policy fragments used by the three `create_*_guardrail.py` scripts:

```python
contentPolicyConfig={"filtersConfig": [
    {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
    {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"},
]},
sensitiveInformationPolicyConfig={"piiEntitiesConfig": [
    {"type": "NAME", "action": "ANONYMIZE"},
    {"type": "EMAIL", "action": "ANONYMIZE"},
]},
```

The S-5 result is the most instructive: the guardrail intervened on the output rather than the input, so the
model saw the full letter and wrote a normal summary, and only the returned text was masked.

![Hate-speech content filter blocking the input](screenshots/S-4-content-blocking.png)

![PII masking replacing names and email in the model output](screenshots/S-5-pii-masking.png)

![Prompt attack filter blocking a role-play jailbreak](screenshots/S-6-prompt-attack-blocked.png)

## Bonus labs

### Lab O-1: Prompt showcase

A single page that pairs a library of prompt templates with a library of sample inputs. Choosing "Summarize" and
"Ethics Complaint Email" fills the template and runs it, so the same input can be tried against several
instructions without editing code. The pattern is a prompt-template registry: templates and examples live in a
config file and the app only does substitution.

![Prompt showcase summarizing a complaint email](screenshots/O-1-prompt-showcase.png)

### Lab O-2: Text generation playground

A free-text box with temperature, top P, and response-token sliders wired straight to the inference
configuration. The sentiment prompt was run at temperature 0.0, which gives a deterministic classification.
Raising temperature or top P is the quickest way to see how the same prompt drifts.

```python
inferenceConfig={"temperature": temperature, "topP": top_p, "maxTokens": max_tokens}
```

![Text playground with inference parameter sliders](screenshots/O-2-text-playground.png)

## Takeaways

- Start with `converse`; fall back to `invoke_model` only when a model needs its native body format.
- Keep model calls in a library module and the UI in a separate file. Every lab did this and it made the
  guardrail and streaming changes one-line edits.
- Build the guardrail once, version it, and pass the ID everywhere. Always enable the trace in development;
  the final text alone does not tell you why a request was blocked or masked.
- Embeddings search is the building block to reach for before a full RAG stack. If ranked passages already
  answer the question, the generation step is optional.
